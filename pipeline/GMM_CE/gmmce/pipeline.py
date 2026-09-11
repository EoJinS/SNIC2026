"""
Top-level orchestration: fit every estimator variant from Sec. IV of the
paper on a training set, and evaluate normalized MSE / component counts
on a test set at a given SNR. This module ties together channel_model,
pilots, complex_gmm, kron_combine, cascade and pdp_ds into the single
interface used by the Fig. 2/3/4 reproduction scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .channel_model import SystemConfig
from .pilots import PilotGrid, vec
from .complex_gmm import GMM, fit_full_gmm, fit_toeplitz_gmm, fit_circulant_gmm
from .kron_combine import combine_kronecker
from .cascade import Cascade2x1D
from .cme import cme_estimate, responsibilities_only, mean_components_for_responsibility
from .pdp_ds import pdp_ds_kron_estimate, pdp_ds_2x1d_estimate

JOINT_VARIANTS = ("full", "b-toep", "b-circ", "kron")
CASCADE_VARIANTS = ("2x1D", "2x1D-toep", "2x1D-circ")
ALL_GMM_VARIANTS = JOINT_VARIANTS + CASCADE_VARIANTS
PDP_DS_VARIANTS = ("PDP+DS kron", "PDP+DS 2x1D")


@dataclass
class TrainedModels:
    cfg: SystemConfig
    grid: PilotGrid
    K: int
    Kt: int
    Kc: int
    Kt_2x1d: int
    Kc_2x1d: int
    gmm_full: GMM = None
    gmm_btoep: GMM = None
    gmm_bcirc: GMM = None
    gmm_time: GMM = None       # unconstrained, dim Nt, Kt comps (for kron / 2x1D)
    gmm_freq: GMM = None       # unconstrained, dim Nc, Kc comps (for kron / 2x1D)
    gmm_time_2x1d: GMM = None  # unconstrained, dim Nt, Kt_2x1d comps (for plain 2x1D)
    gmm_freq_2x1d: GMM = None  # unconstrained, dim Nc, Kc_2x1d comps (for plain 2x1D)
    gmm_time_toep: GMM = None  # dim Nt, Kt comps, Toeplitz-constrained (2x1D-toep)
    gmm_freq_toep: GMM = None  # dim Nc, Kc comps, Toeplitz-constrained (2x1D-toep)
    gmm_time_circ: GMM = None  # dim Nt, Kt comps, circulant (2x1D-circ)
    gmm_freq_circ: GMM = None  # dim Nc, Kc comps, circulant (2x1D-circ)
    gmm_kron: GMM = None       # combined Kt*Kc joint GMM


def train_all(H_train: np.ndarray, grid: PilotGrid, cfg: SystemConfig,
              K: int, Kt: int, Kc: int, Kt_2x1d: int, Kc_2x1d: int, n_iter: int = 20, seed: int = 0,
              which: tuple = ALL_GMM_VARIANTS, n_iter_toep: int | None = None) -> TrainedModels:
    """Fit every requested GMM variant. `which` controls which are fit
    (useful to skip expensive ones during quick experiments).

    Kt/Kc size the 1D GMMs combined multiplicatively for GMM kron
    (Kt*Kc total components); Kt_2x1d/Kc_2x1d size the 1D GMMs used
    additively by the GMM 2x1D cascade (and its toep/circ variants),
    matching the paper's practice of choosing different (Kt, Kc) pairs
    for the two estimators so that Kt*Kc and Kt_2x1d+Kc_2x1d both equal
    the same total component budget (paper Sec. V: kron uses (8, 16),
    2x1D uses (32, 96), both totalling "128 components").

    `n_iter_toep` (default max(3*n_iter, 120)): iteration cap for the
    Toeplitz variants ONLY.  The Barton & Fuhrmann M-step (paper Eq. 6)
    is a single gradient-ascent step on the circulant-embedding spectrum
    per EM iteration -- its log-likelihood rises ~linearly and needs
    ~an order of magnitude more iterations than the closed-form
    sample-covariance M-step of full / kron / circ to converge,
    especially on a sharply-structured (ray-traced) channel covariance
    that is far from the flat-spectrum initialisation.  Under-iterating
    it leaves a biased covariance whose bias is masked by noise at low
    SNR but shows as an NMSE floor at high SNR.  (The tol-based early
    stop inside fit_toeplitz_gmm ends it sooner when it does converge.)"""
    if n_iter_toep is None:
        n_iter_toep = max(3 * n_iter, 120)
    m = TrainedModels(cfg=cfg, grid=grid, K=K, Kt=Kt, Kc=Kc, Kt_2x1d=Kt_2x1d, Kc_2x1d=Kc_2x1d)
    Xtr = vec(H_train)  # (N, Nc*Nt)

    if "full" in which:
        m.gmm_full = fit_full_gmm(Xtr, K=K, n_iter=n_iter, seed=seed)
    if "b-toep" in which:
        m.gmm_btoep = fit_toeplitz_gmm(Xtr, K=K, dims=[cfg.Nt, cfg.Nc], n_iter=n_iter_toep, seed=seed)
    if "b-circ" in which:
        m.gmm_bcirc = fit_circulant_gmm(Xtr, K=K, dims=[cfg.Nt, cfg.Nc], n_iter=n_iter, seed=seed)

    need_kron = "kron" in which
    need_2x1d = "2x1D" in which
    need_time_freq_toep = "2x1D-toep" in which
    need_time_freq_circ = "2x1D-circ" in which

    # H_train: (N, Nc, Nt). A "row" at fixed carrier c is H[:, c, :] (Nt,),
    # used to fit the time-domain GMM; a "column" at fixed time t is
    # H[:, :, t] (Nc,), used to fit the frequency-domain GMM.
    if need_kron or need_2x1d or need_time_freq_toep or need_time_freq_circ:
        time_data = H_train.transpose(1, 0, 2).reshape(-1, cfg.Nt)  # (Nc*N, Nt)
        freq_data = H_train.transpose(2, 0, 1).reshape(-1, cfg.Nc)  # (Nt*N, Nc)

    if need_kron:
        m.gmm_time = fit_full_gmm(time_data, K=Kt, n_iter=n_iter, seed=seed)
        m.gmm_freq = fit_full_gmm(freq_data, K=Kc, n_iter=n_iter, seed=seed)
    if need_2x1d:
        m.gmm_time_2x1d = fit_full_gmm(time_data, K=Kt_2x1d, n_iter=n_iter, seed=seed)
        m.gmm_freq_2x1d = fit_full_gmm(freq_data, K=Kc_2x1d, n_iter=n_iter, seed=seed)
    if need_time_freq_toep:
        m.gmm_time_toep = fit_toeplitz_gmm(time_data, K=Kt_2x1d, dims=[cfg.Nt], n_iter=n_iter_toep, seed=seed)
        m.gmm_freq_toep = fit_toeplitz_gmm(freq_data, K=Kc_2x1d, dims=[cfg.Nc], n_iter=n_iter_toep, seed=seed)
    if need_time_freq_circ:
        m.gmm_time_circ = fit_circulant_gmm(time_data, K=Kt_2x1d, dims=[cfg.Nt], n_iter=n_iter, seed=seed)
        m.gmm_freq_circ = fit_circulant_gmm(freq_data, K=Kc_2x1d, dims=[cfg.Nc], n_iter=n_iter, seed=seed)

    if "kron" in which:
        m.gmm_kron = combine_kronecker(m.gmm_time, m.gmm_freq, Xtr)

    return m


def add_noise(y: np.ndarray, sigma2: float, rng: np.random.Generator) -> np.ndarray:
    return y + (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)) * np.sqrt(sigma2 / 2)


def normalized_mse(H_hat: np.ndarray, H_true: np.ndarray) -> float:
    Nc, Nt = H_true.shape[1], H_true.shape[2]
    return float(np.mean(np.sum(np.abs(H_hat - H_true) ** 2, axis=(1, 2))) / (Nc * Nt))


def evaluate_joint(gmm: GMM, grid: PilotGrid, H_test: np.ndarray, sigma2: float,
                    rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Evaluate a jointly-fit GMM (full/b-toep/b-circ/kron). Returns
    (normalized_mse, H_hat)."""
    y_clean = grid.observe(H_test)  # (N, Np)
    y_noisy = add_noise(y_clean, sigma2, rng).T  # (Np, N)
    Cn = sigma2 * np.eye(grid.Np)
    h_hat = cme_estimate(gmm, grid.A, Cn, y_noisy)  # (D, N)
    from .pilots import unvec
    H_hat = unvec(h_hat.T, grid.Nc, grid.Nt)
    mse = normalized_mse(H_hat, H_test)
    return mse, H_hat


def evaluate_cascade(gmm_freq: GMM, gmm_time: GMM, grid: PilotGrid,
                      H_calib: np.ndarray, H_test: np.ndarray, sigma2: float,
                      rng: np.random.Generator) -> tuple[float, np.ndarray]:
    cascade = Cascade2x1D(gmm_freq, gmm_time, grid)
    cascade.calibrate(H_calib, sigma2)
    true_pilots = H_test[:, grid.pilot_carriers][:, :, grid.pilot_symbols]
    y_noisy = add_noise(true_pilots, sigma2, rng)
    H_hat = cascade.estimate(y_noisy, sigma2)
    mse = normalized_mse(H_hat, H_test)
    return mse, H_hat
