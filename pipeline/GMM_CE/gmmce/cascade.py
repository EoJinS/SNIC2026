"""
Cascaded 1D ("2x1D") channel estimation, Sec. IV-D of the paper: a full
2D time-frequency estimate is produced by first running a 1D
frequency-domain GMM-CME independently at each pilot time-symbol, and
then running a 1D time-domain GMM-CME independently at each carrier,
using the stage-1 outputs in place of pilots.

The paper does not spell out how the two stages' uncertainty should be
combined; we use the common, defensible approximation of treating the
stage-1 estimates as noisy "pilots" for stage 2, with an effective noise
variance calibrated empirically (from a held-out calibration set at the
same SNR) to match the true residual error of stage 1. This is
documented in the README as a modeling choice.
"""
from __future__ import annotations

import numpy as np

from .complex_gmm import GMM
from .cme import cme_estimate
from .pilots import PilotGrid


def _selection_1d(full_dim: int, idx: np.ndarray) -> np.ndarray: # 전체 벡터(1D)에서 특정 index들에 해당하는 애들만 뽑아내는 함수
    A = np.zeros((len(idx), full_dim))
    for i, j in enumerate(idx):
        A[i, j] = 1.0
    return A


class Cascade2x1D:
    """Cascaded 1D estimator built from a frequency-domain GMM (dim Nc)
    and a time-domain GMM (dim Nt)."""

    def __init__(self, gmm_freq: GMM, gmm_time: GMM, grid: PilotGrid):
        self.gmm_freq = gmm_freq
        self.gmm_time = gmm_time
        self.grid = grid
        self.Ac = _selection_1d(grid.Nc, grid.pilot_carriers)   # (Npc, Nc)
        self.At = _selection_1d(grid.Nt, grid.pilot_symbols)    # (Npt, Nt)
        self._Cn_eff = None

    def _stage1(self, H_noisy_pilots: np.ndarray, sigma2: float) -> np.ndarray: # 주파수축 방향 GMM로 주파수방향으로 CME 추정해서 h얻음. 
        """H_noisy_pilots: (N, Npc, Npt) -> stage-1 full-band estimate
        H_stage1: (N, Nc, Npt)."""
        N = H_noisy_pilots.shape[0]
        Npt = self.grid.Npt
        Cn1 = sigma2 * np.eye(self.grid.Npc)
        out = np.empty((N, self.grid.Nc, Npt), dtype=np.complex128)
        for j in range(Npt):
            y = H_noisy_pilots[:, :, j].T  # (Npc, N)
            est = cme_estimate(self.gmm_freq, self.Ac, Cn1, y)  # (Nc, N) 
            out[:, :, j] = est.T
        return out                                               # (N, Nc, Npt)

    def calibrate(self, H_calib: np.ndarray, sigma2: float, seed: int = 0) -> None: # 정답 H의 데이터와 stage1(주파수방향 CME)의 차이로 Cn_eff 구하기.
        """Estimate the effective stage-2 noise variance from the stage-1
        residual error on a calibration set (same SNR as will be used)."""
        rng = np.random.default_rng(seed)
        grid = self.grid
        N = H_calib.shape[0]
        true_pilots = H_calib[:, grid.pilot_carriers][:, :, grid.pilot_symbols]
        noise = (rng.normal(size=true_pilots.shape) + 1j * rng.normal(size=true_pilots.shape)) * np.sqrt(sigma2 / 2)
        H_stage1 = self._stage1(true_pilots + noise, sigma2)  # (N, Nc, Npt)
        true_at_pilot_symbols = H_calib[:, :, grid.pilot_symbols]  # (N, Nc, Npt)
        residual = H_stage1 - true_at_pilot_symbols
        self._Cn_eff = float(np.mean(np.abs(residual) ** 2))

    def estimate(self, H_noisy_pilots: np.ndarray, sigma2: float) -> np.ndarray: # 시간축 방향 GMM로 시간방향으로 CME 추정해서 h얻음. 
        """H_noisy_pilots: (N, Npc, Npt) noisy pilot observations.
        Returns H_hat: (N, Nc, Nt)."""
        if self._Cn_eff is None:
            raise RuntimeError("call calibrate() first")
        N = H_noisy_pilots.shape[0]
        grid = self.grid
        H_stage1 = self._stage1(H_noisy_pilots, sigma2)         # (N, Nc, Npt)

        Cn2 = self._Cn_eff * np.eye(grid.Npt)
        H_hat = np.empty((N, grid.Nc, grid.Nt), dtype=np.complex128)
        for c in range(grid.Nc):
            y = H_stage1[:, c, :].T                             # (Npt, N) 1차 추정결과를 토대로 시간영역 CME 수행
            est = cme_estimate(self.gmm_time, self.At, Cn2, y)  # (Nt, N) 
            H_hat[:, c, :] = est.T
        return H_hat                                            # (N, Nc, Nt)
