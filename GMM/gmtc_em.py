"""
gmtc_em.py — 자립형 GMTC full-covariance GMM EM (공유용, 외부경로 의존 없음)
============================================================================
GMTC main.py 에서 EM 클러스터링에 필요한 함수만 추출한 모듈. torch 만 있으면 됨.
포함: RealGMMParams, _log_gaussian_fullcov, gmm_em_fit_fullcov, predict_modes
(원본과 동일 구현 — /home/ym/GMTC/main.py 에서 추출)
"""
import math
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


# ==== 아래는 GMTC main.py 에서 추출한 원본 코드 ====
@dataclass
class RealGMMParams:
    pi:     torch.Tensor    # (K,)
    mu:     torch.Tensor    # (K, N)
    U:      torch.Tensor    # (K, N, N) orthonormal
    lam:    torch.Tensor    # (K, N) eigenvalues (positive)


def _log_gaussian_fullcov(
    X: torch.Tensor,        # (T, N)
    mu: torch.Tensor,       # (N,)
    Sigma: torch.Tensor,    # (N, N)
    eps: float,
) -> torch.Tensor:
    """
    Compute log N(X | mu, Sigma) for full covariance, in a numerically stable way.

    Uses Cholesky factorization:
        Sigma = L L^T
        quad = || L^{-1} (x - mu)^T ||^2
        logdet = 2 * sum(log(diag(L)))

    Returns:
        logp: (T,) tensor
    """
    T, N = X.shape
    I = torch.eye(N, device=X.device, dtype=X.dtype)

    # Ensure PSD
    Sigma_reg = Sigma + eps * I

    # Cholesky can fail if Sigma_reg is not PD enough; increase jitter if needed.
    jitter = eps
    for _ in range(6):
        try:
            L = torch.linalg.cholesky(Sigma_reg)
            break
        except RuntimeError:
            jitter *= 10.0
            Sigma_reg = Sigma + jitter * I
    else:
        # Last resort: add larger jitter
        L = torch.linalg.cholesky(Sigma + (jitter * 10.0) * I)

    xc = (X - mu)                                               # (T, N)
    y = torch.linalg.solve_triangular(L, xc.T, upper=False)     # (N, T)
    quad = (y * y).sum(dim=0)                                   # (T,)
    logdet = 2.0 * torch.log(torch.diag(L)).sum()               # scalar

    return -0.5 * (quad + logdet + N * math.log(2.0 * math.pi))

def gmm_em_fit_fullcov(
    X: torch.Tensor,             # (T, N) on CPU recommended
    K: int,
    num_iter: int = 50,
    eps: float = 1e-6,
    seed: int = 0,
    verbose: bool = True,       # EM 과정의 log 출력할 때 이전 값과의 비교 필요하면 True
    device: str = None,         # None: use X.device, 'cuda': GPU, 'cpu': CPU
    zero_mean: bool = False,    # True: fix all component means to 0 throughout EM
) -> RealGMMParams:
    """
    Fit a real-valued GMM with full covariance using EM.

    Model:
        p(x) = sum_c pi_c * N(x | mu_c, Sigma_c)

    EM alternates:
        - E-step: responsibilities r_{t,c} p(c | x_t)
        - M-step: update (pi, mu, Sigma) using r_{t,c}

    After EM, each Sigma_c is eigendecomposed to match our parametrization:
        Sigma_c = U_c diag(lam_c) U_c^T

    Returns:
        RealGMMParams(pi, mu, U, lam)
    """
    if X.dim() != 2:
        raise ValueError(f"X must be 2D (T, N), got {tuple(X.shape)}")

    torch.manual_seed(seed)

    T, N = X.shape
    dtype = X.dtype

    # Device handling: use specified device or fall back to X.device
    if device is None:
        compute_device = X.device
    else:
        compute_device = torch.device(device)

    # Move data to compute device
    X_compute = X.to(compute_device)

    # ----- initialization -----
    # mixture weights
    pi = torch.full((K,), 1.0 / K, device=compute_device, dtype=dtype)      # 모든 mode의 weight는 uniform하게 설정

    # means: pick K random samples
    # NOTE: randperm must be generated on CPU for deterministic results across devices
    perm = torch.randperm(T)[:K]  # CPU에서 생성하여 GPU/CPU 동일 결과 보장
    if zero_mean:
        mu = torch.zeros((K, N), device=compute_device, dtype=dtype)
    else:
        mu = X_compute[perm.to(compute_device)].clone()                     # (K, N) - mode별 평균은 training dataset에서 랜덤하게 K개 선택

    # global covariance
    X0 = X_compute - X_compute.mean(dim=0, keepdim=True)                    # covariance 구할 때는 전체 평균을 활용해야 함
    Sigma0 = (X0.T @ X0) / max(T, 1)
    Sigma0 = Sigma0 + eps * torch.eye(N, device=compute_device, dtype=dtype)

    if zero_mean and K > 1:
        # zero_mean에서는 mu가 모두 0으로 동일하므로 Sigma 초기화에 perturbation 필요
        # 각 컴포넌트에 랜덤 서브셋의 sample covariance로 초기화 (symmetry breaking)
        torch.manual_seed(seed + 1)
        Sigma = torch.zeros((K, N, N), device=compute_device, dtype=dtype)
        subset_size = max(N * 2, T // K)
        for c in range(K):
            idx_c = torch.randperm(T, device=compute_device)[:subset_size]
            Xc = X_compute[idx_c]
            Sigma_c = (Xc.T @ Xc) / subset_size
            Sigma[c] = Sigma_c + eps * torch.eye(N, device=compute_device, dtype=dtype)
    else:
        Sigma = Sigma0.unsqueeze(0).repeat(K, 1, 1).contiguous()    # (K, N, N) - mode별 covariance는 전체 dataset에 대한 covariance를 모든 mode에 동일하게 적용

    prev_ll = None

    for it in range(num_iter):
        # ===== E-step =====
        logp = torch.empty((T, K), device=compute_device, dtype=dtype)
        for c in range(K):
            logp[:, c] = torch.log(pi[c] + 1e-30) + _log_gaussian_fullcov(X_compute, mu[c], Sigma[c], eps=eps)

        log_norm = torch.logsumexp(logp, dim=1, keepdim=True)       # (T, 1)
        resp = torch.exp(logp - log_norm)                           # (T, K)

        ll = float(log_norm.sum().item())                           # 근데 이 값 왜 확인하는 거지? 물론 이게 1이 아닐수도 있긴 한데, 결국 iteration 반복되면 특정 값으로 수렴하나?

        if verbose:
            if prev_ll is None:
                print(f"[EM] it={it:03d} ll={ll:.3f}")
            else:
                print(f"[EM] it={it:03d} ll={ll:.3f} dLL={ll - prev_ll:.3f}")
        prev_ll = ll

        # ===== M-step =====
        Nk = resp.sum(dim=0).clamp_min(1e-12)                       # (K,) - mode별 유효 sample 수
        pi = (Nk / T).to(dtype)                                     # mixture weight 업데이트

        if not zero_mean:
            mu = (resp.T @ X_compute) / Nk.unsqueeze(1)             # (K, N) - sample별 resp 이용해서 모든 sample에 대해 mode별 mu 업데이트

        for c in range(K):
            xc = X_compute - mu[c]                                  # (T, N) - 업데이트된 mu로 계산
            w = resp[:, c].unsqueeze(1)                             # (T, 1) - sample별로는 resp가 다르지만, xc의 각 성분에 곱해져야하는 resp는 상수라서 이렇게 차원 넓힌 듯?
            # Weighted covariance: (xc^T diag(w) xc) / Nk
            Sigma_c = (xc * w).T @ xc / Nk[c]
            Sigma[c] = Sigma_c + eps * torch.eye(N, device=compute_device, dtype=dtype)

    # Convert Sigma -> (U, lam)
    U = torch.empty((K, N, N), device=compute_device, dtype=dtype)
    lam = torch.empty((K, N), device=compute_device, dtype=dtype)

    for c in range(K):
        eigvals, eigvecs = torch.linalg.eigh(Sigma[c])              # ascending
        lam[c] = eigvals.clamp_min(eps)
        U[c] = eigvecs

    # Move results back to CPU for compatibility with rest of pipeline
    return RealGMMParams(
        pi=pi.cpu(),
        mu=mu.cpu(),
        U=U.cpu(),
        lam=lam.cpu()
    )

@torch.no_grad()
def predict_modes(
    x: torch.Tensor,
    params: RealGMMParams,
    return_logp: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Oracle MAP mode prediction for real GMM.

    Args:
        x: (B, N) real tensor.
        params: RealGMMParams containing true pi, mu, U, lam.
        return_logp: if True, also return (B, K) log posterior up to an additive constant.

    Returns:
        c_hat: (B,) long tensor, MAP component indices.
        logp: (B, K) tensor if return_logp else None.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D (B, N), got shape {tuple(x.shape)}")

    B, N = x.shape
    K = params.pi.numel()

    # Ensure everything is on the same device/dypte as x
    device = x.device
    dtype = x.dtype

    pi  = params.pi.to(device=device, dtype=dtype)          # (K,)
    mu  = params.mu.to(device=device, dtype=dtype)          # (K, N)
    U   = params.U.to(device=device, dtype=dtype)           # (K, N, N)
    lam = params.lam.to(device=device, dtype=dtype)         # (K, N)

    # Constant term N*log(2*pi) cancels across components for argmax, so omit.
    # Precompute logdet terms per component: log det \Sigma_c = sum_i log lam_{c, i}
    logdet = torch.log(lam).sum(dim=1)                      # (K,)

    logp_list = []
    for c in range(K):
        xc = x - mu[c]                                      # (B, N)
        # Project to eigen-basis (Uc columns are eigenvectors)
        y = xc @ U[c]                                       # (B, N)
        quad = (y * y / lam[c]).sum(dim=1)                  # (B,)
        logp_c = torch.log(pi[c] + 1e-30) - 0.5 * (quad + logdet[c])
        logp_list.append(logp_c)

    logp_all = torch.stack(logp_list, dim=1)                # (B, K)
    c_hat = torch.argmax(logp_all, dim=1).to(torch.long)    # (B,)

    if return_logp:
        return c_hat, logp_all
    return c_hat, None
