"""
Real-channel data adapter for the GMM_CE experiments.

Instead of the synthetic doubly-selective generator in ``channel_model.py``
(frequency x time), this feeds the GMM_CE estimators the POSITION-ONLY
ray-traced POSTECH channels used everywhere else in ``Share/`` -- the
``subregion100`` dense 0.25 m grid, x in [-80, 20], y in [-60, 40], a
1xM TX ULA x F=64 subcarriers, one snapshot per position (no mobility, no
time axis -> the "2-D EM map" regime).  ``m`` picks the pool
``train/test_M{m}_F64_subregion100.npz`` (m=4 is the default 1x4 array;
m=16 the 16-element array), ray-traced by
``generate_channels_subregion100_marray.py``.

Axis mapping onto the paper's (Nc, Nt) = (subcarrier, OFDM symbol) layout
--------------------------------------------------------------------------
The paper's two structured axes are "frequency" (delay-stationary ->
Toeplitz) and "time" (Doppler-stationary -> Toeplitz).  Here the channel's
two structured axes are:

    Nc  <-  subcarrier / frequency   (delay-stationary  -> Toeplitz, DFT = delay domain)
    Nt  <-  antenna    / ULA element (angle-stationary   -> Toeplitz, DFT = beam domain)

so ``C = kron(C_ant, C_freq)`` plays the role of the paper's
``kron(C_time, C_freq)``.  The stored vec order is antenna-major /
subcarrier-minor (``h[p].reshape(M, F)``, index = m*F + f), which is
exactly the paper's ``vec`` convention h[t*Nc + c] with t<-antenna,
c<-subcarrier: ``h.reshape(N, M, F).transpose(0, 2, 1)`` -> H[n, c=f, t=m].

Optional decimation (``nc_ds`` / ``nt_ds``) keeps every k-th subcarrier /
antenna -- still uniformly spaced, so both axes stay Toeplitz/circulant --
to shrink D = Nc*Nt for the O(K D^2 N) unconstrained/Toeplitz EM.
"""
from __future__ import annotations

import os

import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Data")

F_FULL = 64


def _reshape(h: np.ndarray, m_full: int, nc_ds: int, nt_ds: int) -> np.ndarray:
    """(N, M*F) stored antenna-major -> (N, Nc, Nt) = (freq', antenna')."""
    H = h.reshape(-1, m_full, F_FULL)               # [n, antenna, subcarrier]
    H = H[:, ::nt_ds, ::nc_ds]                      # decimate antenna / subcarrier
    return np.ascontiguousarray(H.transpose(0, 2, 1)).astype(np.complex128)   # [n, freq', ant']


class SubregionChannels:
    """Loads the subregion100 pools once and serves disjoint train /
    calib / test splits, normalized so E[||vec(H)||^2] = Nc*Nt (the
    paper's channel normalization)."""

    def __init__(self, m: int = 16, nc_ds: int = 1, nt_ds: int = 1, seed: int = 0):
        self.m, self.nc_ds, self.nt_ds = m, nc_ds, nt_ds
        train_npz = os.path.join(_DATA, f"train_M{m}_F64_subregion100.npz")
        test_npz = os.path.join(_DATA, f"test_M{m}_F64_subregion100.npz")
        tr = np.load(train_npz)
        te = np.load(test_npz)
        self._train = _reshape(np.asarray(tr["h"]), m, nc_ds, nt_ds)
        self._test = _reshape(np.asarray(te["h"]), m, nc_ds, nt_ds)
        self.Nc, self.Nt = self._train.shape[1], self._train.shape[2]

        rng = np.random.default_rng(seed)
        self._perm_train = rng.permutation(len(self._train))
        self._perm_test = rng.permutation(len(self._test))

        # one global real scalar from the training pool
        e = np.mean(np.sum(np.abs(self._train.reshape(len(self._train), -1)) ** 2, axis=1))
        self.scale = float(np.sqrt((self.Nc * self.Nt) / e))
        self._train *= self.scale
        self._test *= self.scale

        # test pool is split: first block = test, second block = calib
        self._n_test_reserved = len(self._test) // 2

    def train(self, n: int, offset: int = 0) -> np.ndarray:
        idx = self._perm_train[offset: offset + n]
        if len(idx) < n:
            raise ValueError(f"train pool exhausted: need {n}, have {len(idx)} from offset {offset}")
        return self._train[idx]

    def test(self, n: int) -> np.ndarray:
        idx = self._perm_test[:self._n_test_reserved][:n]
        if len(idx) < n:
            raise ValueError(f"test pool too small: need {n}, have {len(idx)}")
        return self._test[idx]

    def calib(self, n: int) -> np.ndarray:
        idx = self._perm_test[self._n_test_reserved:][:n]
        if len(idx) < n:
            raise ValueError(f"calib pool too small: need {n}, have {len(idx)}")
        return self._test[idx]
