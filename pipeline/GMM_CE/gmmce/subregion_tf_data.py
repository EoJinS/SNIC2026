"""
SISO TIME-FREQUENCY real-channel data adapter for the GMM_CE experiments.

Unlike ``gmmce/subregion_data.py`` (whose two structured axes are
subcarrier x TX-antenna, a proxy for the paper's subcarrier x OFDM-symbol
layout), this loads the SISO ``Nc(subcarrier) x Nt(OFDM symbol)`` pools
ray-traced by ``generate_channels_subregion100_siso_tf.py`` -- Fesl et
al.'s OWN axis convention, on the same POSTECH ``subregion100`` scene, at
a fixed UE speed (default 10 m/s) with random heading per position.

The stored vec order is already OFDM-symbol-major / subcarrier-minor
(``h[t*F+f]``), i.e. exactly ``gmmce/pilots.py``'s flat ``vec(H)`` index
``h[t*Nc+c]`` with ``Nt <- symbol``, ``Nc <- freq``. Reconstructing the
``(N, Nc, Nt)`` array from that flat layout still needs the same
reshape+transpose ``subregion_data.py`` applies (raw storage is
axis-order ``(symbol, freq)``, the library wants ``(freq, symbol)`` =
``(Nc, Nt)``) -- so despite the friendlier flat-index match, this adapter
is structurally identical to ``SubregionChannels``, just with
``(symbol, freq)`` in place of ``(antenna, freq)`` as the raw storage axes.
"""
from __future__ import annotations

import os

import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Data")


def _reshape(h: np.ndarray, s_full: int, f_full: int, nc_ds: int, nt_ds: int) -> np.ndarray:
    """(N, S*F) stored OFDM-symbol-major -> (N, Nc, Nt) = (freq', symbol')."""
    H = h.reshape(-1, s_full, f_full)                # [n, symbol, subcarrier]
    H = H[:, ::nt_ds, ::nc_ds]                        # decimate symbol / subcarrier
    return np.ascontiguousarray(H.transpose(0, 2, 1)).astype(np.complex128)   # [n, freq', symbol']


class SubregionChannelsTF:
    """Loads the SISO time-frequency subregion100 pools once and serves
    disjoint train / calib / test splits, normalized so
    E[||vec(H)||^2] = Nc*Nt (the paper's channel normalization)."""

    def __init__(self, n_symbols: int = 14, n_subcarriers: int = 64, speed: int = 10,
                 nc_ds: int = 1, nt_ds: int = 1, seed: int = 0, data_tag: str = ""):
        self.n_symbols, self.n_subcarriers, self.speed = n_symbols, n_subcarriers, speed
        tag = f"_{data_tag}" if data_tag else ""
        stem = f"siso_S{n_symbols}_F{n_subcarriers}_v{speed}_subregion100{tag}"
        train_npz = os.path.join(_DATA, f"train_{stem}.npz")
        test_npz = os.path.join(_DATA, f"test_{stem}.npz")
        tr = np.load(train_npz)
        te = np.load(test_npz)
        self._train = _reshape(np.asarray(tr["h"]), n_symbols, n_subcarriers, nc_ds, nt_ds)
        self._test = _reshape(np.asarray(te["h"]), n_symbols, n_subcarriers, nc_ds, nt_ds)
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
