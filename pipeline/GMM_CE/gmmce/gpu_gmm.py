"""
GPU (PyTorch) re-implementations of the complex-GMM EM fits in
``complex_gmm.py``.  Same signatures, same ``GMM`` return type (numpy
arrays), same math -- just the per-component E/M-step linear algebra
batched over K on the GPU (``torch.linalg.cholesky`` / ``cholesky_solve``
/ ``inv`` accept complex tensors and a leading batch dim).

k-means++ seeding stays on the CPU with the numpy Generator so the
initialisation is bit-identical to the CPU path for a given seed.

    from gmmce import gpu_gmm; gpu_gmm.patch()   # monkeypatches pipeline + kron_combine

Env: needs torch with CUDA (the sionna-rt / CSI env has torch 2.2 + cu121).
``GMMCE_GPU_DTYPE=complex64`` trades a little accuracy for ~5-10x speed.
``GMMCE_GPU_DEVICE=cuda:1`` picks the device.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .complex_gmm import GMM, _kmeanspp_init
from .structured import block_operator

_DEV = os.environ.get("GMMCE_GPU_DEVICE", "cuda")
# default precision for GMM full / kron (their near-singular dense-cov LMMSE
# loses ~1.3 dB in fp32); GMMCE_GPU_DTYPE=complex64 forces fp32 everywhere.
_DT = torch.complex64 if os.environ.get("GMMCE_GPU_DTYPE", "complex128") == "complex64" else torch.complex128
# b-toep / diagonal(circ) are fp32-robust (structured constraint regularises)
# and are the runtime bottleneck -> run them in complex64 by default even
# when full/kron stay complex128.  GMMCE_GPU_STRUCT_DTYPE=complex128 to disable.
_DT_STRUCT = torch.complex128 if os.environ.get("GMMCE_GPU_STRUCT_DTYPE", "complex64") == "complex128" else torch.complex64
if _DT == torch.complex64:
    _DT_STRUCT = torch.complex64
_LOG_PI = float(np.log(np.pi))


class _precision:
    """Temporarily switch the module compute dtype (b-toep / diagonal are
    fp32-robust and are the runtime bottleneck, so they run in complex64
    even when full/kron stay complex128)."""
    def __init__(self, dt):
        self.dt = dt

    def __enter__(self):
        global _DT
        self._save = _DT
        _DT = self.dt

    def __exit__(self, *a):
        global _DT
        _DT = self._save


def _rt():
    return torch.float32 if _DT == torch.complex64 else torch.float64


def _t(a):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=_DT, device=_DEV)


def _eye(D):
    return torch.eye(D, dtype=_DT, device=_DEV)


def _herm(C):
    """Hermitian-symmetrise (kills roundoff asymmetry).  C (...,D,D)."""
    return 0.5 * (C + C.conj().transpose(-1, -2))


def _safe_chol(A):
    """Batched lower Cholesky with adaptive diagonal jitter -- the
    ray-traced channel covariance is near-singular (eff. rank ~25/256),
    so a fixed reg is not always enough for cuSOLVER."""
    D = A.shape[-1]
    I = _eye(D)
    scale = torch.diagonal(A, dim1=-2, dim2=-1).real.mean(-1).clamp_min(1e-30)   # (K,)
    A = _herm(A)
    for p in range(-9, 1):                                             # jitter 1e-9 ... 1e0 * trace/D
        L, info = torch.linalg.cholesky_ex(A)
        bad = info != 0
        if not bool(bad.any()):
            return L
        A = A + (bad.to(scale.dtype) * scale * (10.0 ** p)).view(-1, 1, 1) * I
    return torch.linalg.cholesky(_herm(A) + (scale * 10.0).view(-1, 1, 1) * I)


def _batch_logdens(X, means, covs, reg):
    """log CN(x; mean_k, cov_k) for every k and every row of X.
    X (N,D) complex; means (K,D); covs (K,D,D).  Returns (K,N) real.

    Loops over K (not a (K,N,D) tensor) so it survives K=128, N=1e5."""
    K, D = means.shape
    N = X.shape[0]
    L = _safe_chol(covs + reg * _eye(D))                               # (K,D,D)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).real).sum(-1)   # (K,)
    out = torch.empty((K, N), dtype=_rt(), device=X.device)
    for k in range(K):
        Xc = (X - means[k]).transpose(-1, -2)                           # (D,N)
        S = torch.cholesky_solve(Xc, L[k])                              # (D,N)
        out[k] = (-D * _LOG_PI) - logdet[k] - (Xc.conj() * S).sum(0).real
    return out


def _estep(X, log_w, means, covs, reg):
    logp = _batch_logdens(X, means, covs, reg) + log_w.unsqueeze(1)     # (K,N)
    ln = torch.logsumexp(logp, dim=0, keepdim=True)                     # (1,N)
    resp = torch.exp(logp - ln)                                         # (K,N)
    return resp, float(ln.sum().item())


def _mstep_mean_weight(X, resp):
    N = X.shape[0]
    Nk = resp.sum(1) + 1e-12                                # (K,)
    weights = Nk / N
    means = (resp.to(_DT) @ X) / Nk.unsqueeze(1).to(_DT)    # (K,D)
    return weights, Nk, means


def _weighted_scatter(Xc, w):
    """(Xc * w).T @ Xc.conj() -- matches complex_gmm.py's cov convention
    exactly (Hermitian; note it is conj() of the transpose form)."""
    return (Xc.transpose(-1, -2) * w.to(_DT)) @ Xc.conj()


def _mstep_cov(X, resp, means, Nk, reg):
    K, D = means.shape
    covs = torch.empty((K, D, D), dtype=_DT, device=X.device)
    for k in range(K):
        covs[k] = _weighted_scatter(X - means[k], resp[k]) / Nk[k].to(_DT)
    return _herm(covs) + reg * _eye(D)


def _converged(ll, prev, tol):
    return abs(ll - prev) < tol * abs(prev) + 1e-8


def fit_full_gmm(X, K, n_iter=25, reg=1e-6, seed=None, verbose=False, tol=1e-4):
    rng = np.random.default_rng(seed)
    means_np = _kmeanspp_init(X, K, rng)
    Xg = _t(X)
    N, D = Xg.shape
    gcov = np.cov(X.T) if D > 1 else np.array([[np.var(X)]])
    means = _t(means_np)
    covs = _t(gcov + reg * np.eye(D)).unsqueeze(0).repeat(K, 1, 1).contiguous()
    log_w = torch.full((K,), np.log(1.0 / K), dtype=_rt(), device=_DEV)

    prev = -np.inf
    for it in range(n_iter):
        resp, ll = _estep(Xg, log_w, means, covs, reg)
        weights, Nk, means = _mstep_mean_weight(Xg, resp)
        covs = _mstep_cov(Xg, resp, means, Nk, reg)
        log_w = torch.log(weights + 1e-300)
        if verbose:
            print(f"[gpu-full] iter {it}: avg ll = {ll / N:.4f}")
        if _converged(ll, prev, tol):
            break
        prev = ll
    return GMM(weights.cpu().numpy().astype(np.float64),
               means.cpu().numpy().astype(np.complex128),
               covs.cpu().numpy().astype(np.complex128))


def _struct_precision(f):
    """Run f() in the structured-fit precision (_DT_STRUCT, default fp32)."""
    def wrapper(*a, **kw):
        with _precision(_DT_STRUCT):
            return f(*a, **kw)
    wrapper.__name__ = f.__name__
    return wrapper


@_struct_precision
def fit_diagonal_gmm(X, K, n_iter=25, reg=1e-6, seed=None, verbose=False, tol=1e-4):
    rng = np.random.default_rng(seed)
    means_np = _kmeanspp_init(X, K, rng)
    Xg = _t(X)
    N, D = Xg.shape
    means = _t(means_np)
    var = torch.as_tensor(np.var(X, axis=0).real + reg, dtype=_rt(), device=_DEV).unsqueeze(0).repeat(K, 1).contiguous()
    log_w = torch.full((K,), np.log(1.0 / K), dtype=_rt(), device=_DEV)

    prev = -np.inf
    for it in range(n_iter):
        logp = torch.empty((K, N), dtype=_rt(), device=_DEV)
        logconst = torch.log(np.pi * var).sum(1)                        # (K,)
        for k in range(K):
            d2 = ((Xg - means[k]).abs() ** 2) / var[k]                  # (N,D)
            logp[k] = log_w[k] - logconst[k] - d2.sum(1)
        ln = torch.logsumexp(logp, dim=0, keepdim=True)
        resp = torch.exp(logp - ln)                                     # (K,N)
        ll = float(ln.sum().item())
        Nk = resp.sum(1) + 1e-12
        weights = Nk / N
        means = (resp.to(_DT) @ Xg) / Nk.unsqueeze(1).to(_DT)
        for k in range(K):
            var[k] = (resp[k] @ ((Xg - means[k]).abs() ** 2)) / Nk[k] + reg
        log_w = torch.log(weights + 1e-300)
        if _converged(ll, prev, tol):
            break
        prev = ll
    covs = torch.zeros((K, D, D), dtype=_DT, device=_DEV)
    idx = torch.arange(D, device=_DEV)
    covs[:, idx, idx] = var.to(_DT)
    return GMM(weights.cpu().numpy().astype(np.float64),
               means.cpu().numpy().astype(np.complex128),
               covs.cpu().numpy().astype(np.complex128))


def fit_circulant_gmm(X, K, dims, n_iter=25, reg=1e-6, seed=None, tol=1e-4):
    F = block_operator(dims, kind='dft')
    Fh = F.conj().T
    X_tilde = X @ F.T
    g = fit_diagonal_gmm(X_tilde, K, n_iter=n_iter, reg=reg, seed=seed, tol=tol)
    means = g.means @ Fh.T
    d = np.real(np.diagonal(g.covs, axis1=1, axis2=2))
    D = F.shape[0]
    covs = np.empty((K, D, D), np.complex128)
    for k in range(K):
        covs[k] = (Fh * d[k][None, :]) @ F
    return GMM(g.weights, means, covs)


@_struct_precision
def fit_toeplitz_gmm(X, K, dims, n_iter=25, reg=1e-6, seed=None, verbose=False, tol=1e-4):
    rng = np.random.default_rng(seed)
    Qt_np = block_operator(dims, kind='toeplitz')                       # (fourD, D)
    Qt = _t(Qt_np)
    QtH = Qt.conj().transpose(-1, -2)                                   # (D, fourD)
    fourD = Qt.shape[0]
    means_np = _kmeanspp_init(X, K, rng)
    Xg = _t(X)
    N, D = Xg.shape
    reg_eff = reg if _DT == torch.complex128 else max(reg, 1e-4)        # fp32 needs a bigger floor
    gcov = np.cov(X.T) if D > 1 else np.array([[np.var(X)]], dtype=np.complex128)
    c0 = max(np.real(np.trace(gcov)) / D, reg_eff)
    c = torch.full((K, fourD), float(c0), dtype=_rt(), device=_DEV)
    means = _t(means_np)
    log_w = torch.full((K,), np.log(1.0 / K), dtype=_rt(), device=_DEV)

    def covs_from_c(cc):                                                # (K,fourD) -> (K,D,D)
        return _herm(torch.einsum('df,kf,fe->kde', QtH, cc.to(_DT), Qt)) + reg_eff * _eye(D)

    covs = covs_from_c(c)
    prev = -np.inf
    for it in range(n_iter):
        resp, ll = _estep(Xg, log_w, means, covs, reg_eff)
        weights, Nk, means = _mstep_mean_weight(Xg, resp)
        Cinv = torch.cholesky_inverse(_safe_chol(covs))                 # (K,D,D), robust
        newc = torch.empty_like(c)
        for k in range(K):
            Chat = _weighted_scatter(Xg - means[k], resp[k]) / Nk[k].to(_DT)
            M = Cinv[k] @ Chat @ Cinv[k] - Cinv[k]                      # (D,D)
            Theta = Qt @ M @ QtH                                        # (fourD, fourD)
            step = c[k] * torch.diagonal(Theta).real * c[k]
            newc[k] = torch.clamp(c[k] + step, min=reg_eff)
        c = newc
        covs = covs_from_c(c)
        log_w = torch.log(weights + 1e-300)
        if verbose:
            print(f"[gpu-toep] iter {it}: avg ll = {ll / N:.4f}")
        if _converged(ll, prev, tol):
            break
        prev = ll
    return GMM(weights.cpu().numpy().astype(np.float64),
               means.cpu().numpy().astype(np.complex128),
               covs.cpu().numpy().astype(np.complex128))


def combine_kronecker(gmm_time: GMM, gmm_freq: GMM, X_train, reg=1e-6) -> GMM:
    Kt, Kc = gmm_time.K, gmm_freq.K
    K = Kt * Kc
    D = gmm_time.D * gmm_freq.D
    means = np.empty((K, D), np.complex128)
    covs = np.empty((K, D, D), np.complex128)
    init_w = np.empty(K)
    i = 0
    for kt in range(Kt):
        for kc in range(Kc):
            means[i] = np.kron(gmm_time.means[kt], gmm_freq.means[kc])
            covs[i] = np.kron(gmm_time.covs[kt], gmm_freq.covs[kc])
            init_w[i] = gmm_time.weights[kt] * gmm_freq.weights[kc]
            i += 1
    Xg = _t(X_train)
    logp = _batch_logdens(Xg, _t(means), _t(covs), reg) + torch.log(_t(init_w).real + 1e-300).unsqueeze(1)
    ln = torch.logsumexp(logp, dim=0, keepdim=True)
    resp = torch.exp(logp - ln)
    w = resp.mean(1).cpu().numpy().astype(np.float64)
    w /= w.sum()
    return GMM(w, means, covs)


def cme_estimate(gmm, A, Cn, Y, reg=1e-9, batch_size=4000):
    """GPU GMM-CME (Eq. 2-3), same interface as cme.cme_estimate.
    A (Np,D); Cn (Np,Np); Y (Np, N_test).  Returns h_hat (D, N_test)."""
    K, D = gmm.means.shape
    Ag = _t(A)
    means = _t(gmm.means)
    covs = _t(gmm.covs)
    Cng = _t(Cn)
    log_w = torch.log(torch.as_tensor(gmm.weights, dtype=_rt(), device=_DEV) + 1e-300)
    Np = A.shape[0]
    CAh = covs @ Ag.conj().transpose(-1, -2)                 # (K,D,Np)
    Cy = Ag @ CAh + Cng + reg * torch.eye(Np, dtype=_DT, device=_DEV)     # (K,Np,Np)
    L = _safe_chol(Cy)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).real).sum(-1)   # (K,)
    Amu = means @ Ag.transpose(-1, -2)                       # (K,Np)
    Yg = _t(Y)
    Ntest = Yg.shape[1]
    out = torch.empty((D, Ntest), dtype=_DT, device=_DEV)
    for s in range(0, Ntest, batch_size):
        e = min(s + batch_size, Ntest)
        diff = Yg[:, s:e].unsqueeze(0) - Amu.unsqueeze(2)    # (K,Np,b)
        sol = torch.cholesky_solve(diff, L)                  # (K,Np,b)
        maha = (diff.conj() * sol).sum(1).real              # (K,b)
        loglik = log_w.unsqueeze(1) - Np * _LOG_PI - logdet.unsqueeze(1) - maha
        g = torch.softmax(loglik, dim=0)                     # (K,b)
        est = means.unsqueeze(2) + CAh @ sol                 # (K,D,b)
        out[:, s:e] = torch.einsum('kb,kdb->db', g.to(_DT), est)
    return out.cpu().numpy().astype(np.complex128)


def patch():
    """Redirect pipeline + kron_combine + CME to the GPU implementations."""
    import gmmce.pipeline as pl
    import gmmce.kron_combine as kc
    import gmmce.complex_gmm as cg
    import gmmce.cme as cme_mod
    import gmmce.cascade as cascade_mod
    pl.fit_full_gmm = fit_full_gmm
    pl.fit_toeplitz_gmm = fit_toeplitz_gmm
    pl.fit_circulant_gmm = fit_circulant_gmm
    cg.fit_diagonal_gmm = fit_diagonal_gmm
    pl.combine_kronecker = combine_kronecker
    kc.combine_kronecker = combine_kronecker
    pl.cme_estimate = cme_estimate
    cme_mod.cme_estimate = cme_estimate
    cascade_mod.cme_estimate = cme_estimate
    print(f"[gpu_gmm] patched -- device={_DEV}  full/kron dtype={_DT}  toep/circ dtype={_DT_STRUCT}")
