'''
Requirements:
numpy==1.26.4
scipy==1.15.3
torch==2.2.2
'''

import math
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# ================= GMM 분포 설계 및 데이터셋 생성 =================

def dct_orthonormal_matrix(N: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Orthonormal DCT-II matrix (N x N), real.
    """
    n = torch.arange(N, device=device, dtype=dtype).view(-1, 1)     # (N, 1)
    k = torch.arange(N, device=device, dtype=dtype).view(1, -1)     # (1, N)

    # DCT-II basis
    mat = torch.cos(math.pi / N * (n + 0.5) * k)    # (N, N)

    # Orthonormal scaling
    mat[:, 0] *= math.sqrt(1.0 / N)
    if N > 1:
        mat[:, 1:] *= math.sqrt(2.0 / N)
    return mat

def make_component_U_from_real_basis(B: torch.Tensor, r: int, shift: int) -> torch.Tensor:
    """
    Build U_c by permuting columns of an orthonormal real basis B (e.g., DCT).
    Keeps orthonormality.
    """
    N = B.shape[0]
    dom = [(i + shift) % N for i in range(r)]       # dominant subspace 선택
    rest = [i for i in range(N) if i not in dom]
    perm = dom + rest
    return B[:, perm]

@dataclass
class RealGMMParams:
    pi:     torch.Tensor    # (K,)
    mu:     torch.Tensor    # (K, N)
    U:      torch.Tensor    # (K, N, N) orthonormal
    lam:    torch.Tensor    # (K, N) eigenvalues (positive)

def build_real_gmm_params(
    N: int = 32,
    K: int = 4,
    r: int = 6,
    lam_hi: float = 10.0,
    lam_lo: float = 0.1,
    shifts: Optional[Tuple[int, ...]] = None,
    pi: Optional[torch.Tensor] = None,
    mean_scale: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> RealGMMParams:
    torch.manual_seed(seed)
    device = torch.device(device)

    # shifts - 기본 설정으로 U가 DCT(DFT) matrix인데, mode별로 column shift가 진행되게 설계됐나 봄 - mode별로 shift가 일정하게 더 돌아가도록 설정된 듯
    if shifts is None:
        step = max(1, N // K)
        shifts = tuple((i * step) % N for i in range(K))
    else:
        assert len(shifts) == K

    # pi - GMM 각 mode의 분포인 듯? 기본 설정은 uniform distribution인 거 같고
    if pi is None:
        pi = torch.full((K,), 1.0 / K, device=device, dtype=dtype)
    else:
        pi = pi.to(device=device, dtype=dtype)
        pi = pi / pi.sum()

    # Orthonormal real basis (DCT)
    B = dct_orthonormal_matrix(N, device=device, dtype=dtype)

    # Component-specific orthonormal matrices U_c via column permutation
    U_list = []
    for c in range(K):
        U_list.append(make_component_U_from_real_basis(B, r=r, shift=shifts[c]))
    U = torch.stack(U_list, dim=0)  # (K, N, N)

    # Eigenvalues
    lam = torch.full((K, N), lam_lo, device=device, dtype=dtype)
    lam[:, :r] = lam_hi

    # Means
    if mean_scale == 0.0:
        mu = torch.zeros((K, N), device=device, dtype=dtype)
    else:
        mu = mean_scale * torch.randn((K, N), device=device, dtype=dtype)

    return RealGMMParams(pi=pi, mu=mu, U=U, lam=lam)

@torch.no_grad()
def sample_real_gmm(T: int, params: RealGMMParams, return_labels: bool = True):
    """
    Sample T vectors x in R^N from the Real GMM.

    x = mu_c + z_scaled @ U_c^T, where z_scaled ~ N(0, diag(lam_c))
    (row-vector convention)
    """
    pi = params.pi
    K = pi.shape[0]
    device = pi.device
    dtype = params.mu.dtype
    N = params.mu.shape[1]

    labels = torch.distributions.Categorical(probs=pi).sample((T,)).to(device)
    x = torch.empty((T, N), device=device, dtype=dtype)

    for c in range(K):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        m = idx.numel()
        z = torch.randn((m, N), device=device, dtype=dtype)     # N(0, I)
        z_scaled = z * torch.sqrt(params.lam[c])                # scale per-dimension
        Uc = params.U[c]                                        # (N, N)
        x_c = params.mu[c] + (z_scaled @ Uc.T)                  # (m, N)
        x[idx] = x_c

    return (x, labels) if return_labels else (x, None)

class RealGMMDataset(Dataset):
    """
    Real GMM dataset: returns (x, label) or x only.
    - x: float tensor (N,)
    - label: long tensor scalar
    """
    def __init__(
        self,
        T: int,
        N: int = 32,
        K: int = 4,
        r: int = 6,
        lam_hi: float = 10.0,
        lam_lo: float = 0.1,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        return_labels: bool = True,
        mean_scale: float = 0.0,
        params: Optional["RealGMMParams"] = None,
    ):
        super().__init__()
        self.return_labels = return_labels

        if params is None:
            params = build_real_gmm_params(
                N=N, K=K, r=r,
                lam_hi=lam_hi, lam_lo=lam_lo,
                seed=seed, device=device, dtype=dtype,
                mean_scale=mean_scale
            )
        self.params = params        # 멤버 변수로 나타내어 나중에 TC/KLT할 때 재사용

        # pre-generate
        x, c = sample_real_gmm(T, params, return_labels=True)

        # dataset 저장은 cpu에 두는 게 보통 편함 (DataLoader pin_memory 등)
        self.x = x.detach().cpu()
        self.c = c.detach().cpu()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.return_labels:
            return self.x[idx], self.c[idx]
        return self.x[idx]

def make_gmm_train_test_datasets(
    T_train: int = 20000,
    T_test: int = 20000,
    N: int = 32,
    K: int = 4,
    r: int = 6,
    lam_hi: float = 10.0,
    lam_lo: float = 0.1,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
):
    # train/test가 "같은 분포"에서 나오도록 params 공유
    params = build_real_gmm_params(
        N=N, K=K, r=r, lam_hi=lam_hi, lam_lo=lam_lo,
        seed=seed, device="cpu", dtype=dtype, mean_scale=0.0
    )

    train_ds = RealGMMDataset(
        T=T_train, return_labels=True, params=params
    )
    test_ds = RealGMMDataset(
        T=T_test, return_labels=True, params=params
    )
    return train_ds, test_ds, params


# ================= Oracle 기반 mode 추정 (GMM 파라미터를 알고 있다고 가정) =================
# ================= 나중에는 EM 알고리즘 기반으로 파라미터 추정하는 파트도 추가하기 =================

@torch.no_grad()
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

@torch.no_grad()
def gmm_em_fit_fullcov(
    X: torch.Tensor,             # (T, N) on CPU recommended
    K: int,
    num_iter: int = 50,
    eps: float = 1e-6,
    seed: int = 0,
    verbose: bool = True,       # EM 과정의 log 출력할 때 이전 값과의 비교 필요하면 True
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
    device = X.device
    dtype = X.dtype

    # ----- initialization -----
    # mixture weights
    pi = torch.full((K,), 1.0 / K, device=device, dtype=dtype)      # 모든 mode의 weight는 uniform하게 설정

    # means: pick K random samples
    perm = torch.randperm(T, device=device)[:K]
    mu = X[perm].clone()                                            # (K, N) - mode별 평균은 training dataset에서 랜덤하게 K개 선택

    # global covariance
    X0 = X - X.mean(dim=0, keepdim=True)                            # covariance 구할 때는 전체 평균을 활용해야 함
    Sigma0 = (X0.T @ X0) / max(T, 1)
    Sigma0 = Sigma0 + eps * torch.eye(N, device=device, dtype=dtype)

    Sigma = Sigma0.unsqueeze(0).repeat(K, 1, 1).contiguous()        # (K, N, N) - mode별 covariance는 전체 dataset에 대한 covariance를 모든 mode에 동일하게 적용

    prev_ll = None

    for it in range(num_iter):
        # ===== E-step =====
        logp = torch.empty((T, K), device=device, dtype=dtype)
        for c in range(K):
            logp[:, c] = torch.log(pi[c] + 1e-30) + _log_gaussian_fullcov(X, mu[c], Sigma[c], eps=eps)

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

        mu = (resp.T @ X) / Nk.unsqueeze(1)                         # (K, N) - sample별 resp 이용해서 모든 sample에 대해 mode별 mu 업데이트

        for c in range(K):
            xc = X - mu[c]                                          # (T, N) - 업데이트된 mu로 계산
            w = resp[:, c].unsqueeze(1)                             # (T, 1) - sample별로는 resp가 다르지만, xc의 각 성분에 곱해져야하는 resp는 상수라서 이렇게 차원 넓힌 듯?
            # Weighted covariance: (xc^T diag(w) xc) / Nk
            Sigma_c = (xc * w).T @ xc / Nk[c]
            Sigma[c] = Sigma_c + eps *  torch.eye(N, device=device, dtype=dtype)

    # Convert Sigma -> (U, lam)
    U = torch.empty((K, N, N), device=device, dtype=dtype)
    lam = torch.empty((K, N), device=device, dtype=dtype)

    for c in range(K):
        eigvals, eigvecs = torch.linalg.eigh(Sigma[c])              # ascending
        lam[c] = eigvals.clamp_min(eps)
        U[c] = eigvecs

    return RealGMMParams(pi=pi, mu=mu, U=U, lam=lam)



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



# ================= sample에 대해 mode에 적합한 source coding 적용 =================

# ================= (1) Conditional Unitary Transform =================

@torch.no_grad()
def conditional_transform(
    x: torch.Tensor,
    c_hat: torch.Tensor,
    params: RealGMMParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project samples into the selected mode's eigen-basis.

    For each sample x_b assigned to mode c_hat[b]:
        y_b = (x_b - mu_c) @ U_c

    Returns:
        y: (B, N) eigen-basis coefficients
        lam_sel: (B, N) eigenvalues corresponding to each sample's selected mode
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D (B, N), got {tuple(x.shape)}")
    if c_hat.dim() != 1:
        raise ValueError(f"c_hat must be 1D (B,), got {tuple(c_hat.shape)}")

    B, N = x.shape
    device = x.device
    dtype = x.dtype

    mu  = params.mu.to(device=device, dtype=dtype)          # (K, N)
    U   = params.U.to(device=device, dtype=dtype)           # (K, N, N)
    lam = params.lam.to(device=device, dtype=dtype)         # (K, N)

    y = torch.empty((B, N), device=device, dtype=dtype)
    lam_sel = torch.empty((B, N), device=device, dtype=dtype)


    K = params.pi.numel()
    for c in range(K):
        idx = (c_hat == c).nonzero(as_tuple=True)[0]        # mode c에 해당하는 sample들의 index - m은 그 개수
        if idx.numel() == 0:
            continue

        xc = x[idx] - mu[c]                                 # (m, N)
        y[idx] = xc @ U[c]                                  # (m, N)
        lam_sel[idx] = lam[c].expand(idx.numel(), -1)       # (m, N)

    return y, lam_sel


# ================= (2) Reverse water-filling 기반 theoretical rate 계산 ================= 
   
@torch.no_grad()
def active_mask_from_lam(lam_sel: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Active dimension mask under reverse water-filling rule (lam > gamma).
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    return lam_sel.clamp_min(1e-30) > gamma


# gamma를 sweep하면 (R, D) curve 얻기 가능
# 특정 rate budget에 맞는 gamma를 원하면, R(gamma)가 단조함수라서 이분탐색으로 최적 gamma를 찾을 수 있음.
@torch.no_grad()
def theoretical_rwf_rate(
    lam_sel: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Theoretical Gaussian RD rate allocation (reverse water-filling).
    Compute per-sample rates under reverse water-filling for independent Gaussian dims.

    For each dimension with variance lam_i:
        if lam_i > gamma:   r_i = 0.5 * log2(lam_i / gamma)
        else:               r_i = 0

    Returns:
        r_dim: (B, N) per-dimension rate (bits per dimension)
        r_sum: (B,) total rate per sample (bits per vector)
        active_mask: (B, N) bool mask

    Notes:
        - This is *not* an operational bit-count. It's the Shannon RD allocation for independent Gaussian dims.
        - Operational RD는 아래의 quantization 결과(q)에서 empirical entropy를 계산해서 rate를 근사할 예정.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    lam_sel = lam_sel.clamp_min(1e-30)
    active = lam_sel > gamma

    r = 0.5 * torch.log2(lam_sel / gamma)
    r = torch.where(active, r, torch.zeros_like(r))

    r_sum = r.sum(dim=1)
    return r, r_sum, active


# ================= (3) Operational TC: encode (quantize y) ================= 

@torch.no_grad()
def uniform_mid_tread_quantize(x: torch.Tensor, delta: float) -> torch.Tensor:
    """
    Uniform mid-tread scalar quantizer.
    q = round(x / delta)

    Returns:
        q: int64 tensor with same shape as x
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")
    return torch.round(x / delta).to(torch.int64)

@torch.no_grad()
def tc_encode_y_given(
    y: torch.Tensor,
    lam_sel: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Encoder helper when y and lam_sel are already computed.
    (conditional_transform을 이미 수행한 경우 사용)

    This removes duplicated logic across the pipeline.

    Steps:
        1) active dims: lam_sel > gamma
        2) choose delta so that quantization MSE ~= gamma for real scalar quantization:
            MSE ~= delta^2 / 12 => delta = sqrt(12 * gamma)
        3) quantize active y with uniform mid-tread quantizer; inactive dims are 0

    Returns:
        q: (B, N) int64 indices (inactive dims are 0)
        delta: scalar step size
        active_mask: (B, N) bool
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    
    active = lam_sel > gamma
    delta = float(math.sqrt(12.0 * gamma))

    q = torch.zeros_like(y, dtype=torch.int64)
    if active.any():
        q_active = uniform_mid_tread_quantize(y[active], delta=delta)
        q[active] = q_active

    return q, delta, active


@torch.no_grad()
def tc_encode_y_oracle(
    x: torch.Tensor,
    c_hat: torch.Tensor,
    params: RealGMMParams,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Encoder(oracle): conditional transform -> drop(active mask) -> uniform quantize of y.

    Steps:
        1) (y, lam_sel) = conditional_transform(x, c_hat, params)
        2) active dims: lam_sel > gamma (reverse water-filling style)
        3) quantize active y with uniform mid-tread quantizer
            - choose delta so that quantization MSE ~= gamma for real scalar quantization:
            MSE ~= delta^2 / 12 => delta = sqrt(12 * gamma)
        4) produce integer indices q (inactive dims are 0)

    Returns:
        y: (B, N) eigen-basis coefficients
        q: (B, N) int64 quantization indices (inactive dims are 0)
        delta: scalar step size
        active_mask: (B, N) bool
        lam_sel: (B, N) eigenvalues selected by mode (useful for analysis)
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    y, lam_sel = conditional_transform(x, c_hat, params)
    q, delta, active = tc_encode_y_given(y, lam_sel, gamma=gamma)

    return y, q, delta, active, lam_sel


# ================= (4) Operational TC: decode (reconstruct x) ================= 

@torch.no_grad()
def uniform_mid_tread_dequantize(q: torch.Tensor, delta: float, dtype: torch.dtype) -> torch.Tensor:
    """
    Inverse of uniform_mid_tread_quantize.
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")
    return q.to(dtype) * delta

@torch.no_grad()
def conditional_inverse_transform(
    y_hat: torch.Tensor,
    c_hat: torch.Tensor,
    params: RealGMMParams,
) -> torch.Tensor:
    """
    Inverse of conditional_transform.

    For each sample with mode c:
        x_hat = mu_c + y_hat @ U_c^T

    Args:
        y_hat: (B, N) eigen-basis coefficients (possibly quantized / with dropped dims)
        c_hat: (B,) mode indices

    Returns:
        x_hat: (B, N)
    """
    if y_hat.dim() != 2:
        raise ValueError(f"y_hat must be 2D (B, N), got {tuple(y_hat.shape)}")
    if c_hat.dim() != 1:
        raise ValueError(f"c_hat must be 1D (B,), got {tuple(c_hat.shape)}")

    B, N = y_hat.shape
    device = y_hat.device
    dtype = y_hat.dtype

    mu = params.mu.to(device=device, dtype=dtype)               # (K, N)
    U = params.U.to(device=device, dtype=dtype)                 # (K, N, N)

    x_hat = torch.empty((B, N), device=device, dtype=dtype)
    K = params.pi.numel()
    for c in range(K):
        idx = (c_hat == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        x_hat[idx] = mu[c] + (y_hat[idx] @ U[c].T)

    return x_hat

@torch.no_grad()
def tc_decode_y_oracle(
    q: torch.Tensor,
    c_hat: torch.Tensor,
    params: RealGMMParams,
    delta: float,
    active_mask: torch.Tensor,      # 이거 왜 필요하지? 어차피 q에 0이 들어있으면 masking된 거로 보면 되는 거 아닌가? 아 근데 연산 관점으로 보면 전달해주는 게 좋긴 하겠다. 근데 실제로 구현할 때는 어떻게 되는거지?
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decoder(oracle): dequantize to y_hat and inverse transform to x_hat.

    Args:
        q: (B, N) int64 indices
        c_hat: (B,) mode indices (assumed decoded correctly)
        delta: quantizer step size
        active_mask: (B, N) bool mask (inactive dims treated as 0)
        dtype: output dtype (typically x.dtype)

    Returns:
        y_hat: (B, N) dequantized coefficients (inactive dims are 0)
        x_hat: (B, N) reconstructed vectors
    """
    if q.dim() != 2:
        raise ValueError(f"q must be 2D (B, N), got {tuple(q.shape)}")
    if c_hat.dim() != 1:
        raise ValueError(f"c_hat must be 1D (B,), got {tuple(c_hat.shape)}")

    y_hat = torch.zeros_like(q, dtype=dtype)
    if active_mask.any():
        y_hat[active_mask] = uniform_mid_tread_dequantize(q[active_mask], delta=delta, dtype=dtype)

    x_hat = conditional_inverse_transform(y_hat, c_hat, params)
    return y_hat, x_hat


# ================= mode detection accuracy를 위한 utility =================
@torch.no_grad()
def confusion_matrix(true_c: torch.Tensor, pred_c: torch.Tensor, K: int) -> torch.Tensor:
    """
    Compute KxK confusion matrix where rows=true labels, pred labels.
    """
    true_c = true_c.to(torch.long).reshape(-1)
    pred_c = pred_c.to(torch.long).reshape(-1)
    cm = torch.zeros((K, K), dtype=torch.long)
    for i in range(true_c.numel()):
        cm[true_c[i], pred_c[i]] += 1
    return cm

@torch.no_grad()
def best_label_permutation(true_c: torch.Tensor, pred_c: torch.Tensor, K: int):
    """
    Find the permutation of predicted labels that maximizes accuracy.

    Returns:
        best_acc: float
        best_perm: tuple of length K, where best_perm[p] = mapped label for predicted label p.
        pred_mapped: tensor same shape as pred_c, with labels remapped by best_perm.
    """
    import itertools

    true_c = true_c.to(torch.long).reshape(-1)
    pred_c = pred_c.to(torch.long).reshape(-1)

    best_acc = -1.0
    best_perm = tuple(range(K))

    for perm in itertools.permutations(range(K)):
        perm_t = torch.tensor(perm, dtype=torch.long)
        mapped = perm_t[pred_c]
        acc = (mapped == true_c).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_perm = perm

    perm_t = torch.tensor(best_perm, dtype=torch.long)
    pred_mapped = perm_t[pred_c]
    return best_acc, best_perm, pred_mapped.reshape_as(true_c)



# ================= 성능 평가를 위한 RD curve 관련 함수들 =================

@torch.no_grad()
def mse_per_vector(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Per-sample MSE averaged over dimensions.
    """
    return ((x - x_hat) ** 2).mean(dim=1)

@torch.no_grad()
def empirical_entropy_bits_per_symbol(x: torch.Tensor) -> float:
    """
    Empirical entropy H(X) in bits/symbol for a 1D integer tensor.

    Args:
        x: 1D tensor of integers (e.g., quantizer indices)

    Returns:
        H: float, in bits/symbol
    """
    if x.numel() == 0:
        return 0.0

    # Make contiguous 1D
    x = x.reshape(-1)

    # Map values to [0, ..., M-1] via unique
    vals, counts = torch.unique(x, return_counts=True)
    p = counts.to(torch.float64) / counts.sum().to(torch.float64)
    H = -(p * torch.log2(p)).sum().item()
    return float(H)

@torch.no_grad()
def label_entropy_bits_per_vector(c_hat: torch.Tensor, K: int) -> float:
    """
    Empirical entropy of labels in bits/vector.
    """
    if c_hat.numel() == 0:
        return 0.0
    counts = torch.bincount(c_hat.to(torch.long), minlength=K).to(torch.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    H = -(p * torch.log2(p)).sum().item()
    return float(H)


@torch.no_grad()
def label_entropy_from_pi(pi: torch.Tensor) -> float:
    """
    Oracle label entropy H(c) computed from the true mixture weights pi (bits/vector).
    """
    pi = pi.detach().to(torch.float64)
    pi = pi / pi.sum()
    pi = pi[pi > 0]
    H = -(pi * torch.log2(pi)).sum().item()
    return float(H)


@torch.no_grad()
def theoretical_distortion_from_lam(lam_sel: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Theoretical distortion (per-sample, per-dimension MSE) under reverse water-filling.

    For independent Gaussian dims with variances lam_i and water level gamma,
    the optimal distortion per dimension is min(lam_i, gamma).
    이건 y = U^T (x - \mu) 형태라서 x에 대한 MSE와 y에 대한 MSE가 사실 동일하다는 것을 이용한 사실.

    Returns:
        d_vec: (B,) per-sample distortion averaged over dimensions.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    g = torch.as_tensor(gamma, device=lam_sel.device, dtype=lam_sel.dtype)
    d_dim = torch.minimum(lam_sel, g)
    return d_dim.mean(dim=1)

@torch.no_grad()
def operational_rate_bits_per_vector(
    q: torch.Tensor,
    active_mask: torch.Tensor,
    c_hat: torch.Tensor,
    K: int,
    H_label_override: Optional[float] = None,
) -> float:
    """
    Operational rate estimate in bits/vector using empirical entropy.

    We assume ideal arithmetic coding with *parallel* per-dimension codebooks:
        R \approx H(c_hat) + \sum_{i=1}^N H(q_i)
    where q_i is the quantizer index reandom variable for dimension i.

    Implementation:
        - H(c_hat): empirical label entropy from predicted labels.
        - For each dimension i, compute H(q_i) using only samples where that dimension is active.

    Notes:
        - This is a finite-sample estimate (empirical entropy).
        - You can pass H_label_override=H(pi) (oracle or EM-estimated) to use a fixed label-cost instead of empirical H(c_hat).
    """
    # Move once to CPU to avoid repeated device transfers inside the loop
    c_hat_cpu = c_hat.detach().cpu()
    q_cpu = q.detach().cpu()
    active_cpu = active_mask.detach().cpu()

    if H_label_override is None:
        H_label = label_entropy_bits_per_vector(c_hat_cpu, K=K)
    else:
        H_label = float(H_label_override)

    # Index cost: sum_i H(q_i) over active dimensions (parallel arithmetic coding per dimension)
    B, N = q.shape
    R_idx = 0.0
    for i in range(N):
        mask_i = active_cpu[:, i]
        if mask_i.any():
            q_i = q_cpu[mask_i, i]
            R_idx += empirical_entropy_bits_per_symbol(q_i)

    return float(H_label + R_idx)


# ================= Single Gaussian Baseline =================

@dataclass
class SingleGaussianParams:
    mu: torch.Tensor        # (N,)
    U: torch.Tensor         # (N, N) orthonormal eigenvectors (columns)
    lam: torch.Tensor       # (N,) eigenvalues (positive)

@torch.no_grad()
def fit_single_gaussian_fullcov(X: torch.Tensor, eps: float = 1e-6) -> SingleGaussianParams:
    """
    Fit a single full-covariance Gaussian N(mu, Sigma) from samples X (T, N).
    """
    if X.dim() != 2:
        raise ValueError(f"X must be 2D (T, N), got {tuple(X.shape)}")
    
    T, N = X.shape
    X = X.to(torch.float32)

    mu = X.mean(dim=0)      # (N,)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / max(T, 1)
    Sigma = Sigma + eps * torch.eye(N, device=X.device, dtype=X.dtype)

    eigvals, eigvecs = torch.linalg.eigh(Sigma)     # ascending
    lam = eigvals.clamp_min(eps)
    U = eigvecs

    return SingleGaussianParams(mu=mu, U=U, lam=lam)

@torch.no_grad()
def single_gaussian_transform(x: torch.Tensor, sg: SingleGaussianParams) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    y = (x - mu) @ U, lam_sel is broadcast to (B, N).
    """
    if x.dim() != 2:
        raise ValueError(f"x must be (B, N), got {tuple(x.shape)}")
    
    B, N = x.shape
    mu = sg.mu.to(device=x.device, dtype=x.dtype)
    U = sg.U.to(device=x.device, dtype=x.dtype)
    lam = sg.lam.to(device=x.device, dtype=x.dtype)

    y = (x - mu) @ U
    lam_sel = lam.unsqueeze(0).expand(B, -1)
    return y, lam_sel

@torch.no_grad()
def single_gaussian_inverse_transform(y_hat: torch.Tensor, sg: SingleGaussianParams) -> torch.Tensor:
    """
    x_hat = mu + y_hat @ U^T
    """
    if y_hat.dim() != 2:
        raise ValueError(f"y_hat must be 2D (B, N), got {tuple(y_hat.shape)}")
    
    mu = sg.mu.to(device=y_hat.device, dtype=y_hat.dtype)
    U = sg.U.to(device=y_hat.device,dtype=y_hat.dtype)
    return mu + (y_hat @ U.T)


# ================= Rate-distortion curve 그리기 =================


@torch.no_grad()
def sweep_rd_curves(
    x: torch.Tensor,
    c_hat: Optional[torch.Tensor],
    params: Optional[RealGMMParams],
    gammas: torch.Tensor,
    *,
    # If provided, use this label cost for BOTH theoretical and operational curves.
    # If None: theoretical uses H(pi) from params, operational uses empirical H(c_hat).
    H_label_override: Optional[float] = None,
    # Single-Gaussian baseline
    single_gaussian: Optional[SingleGaussianParams] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sweep gamma values and compute (R, D) for theoretical and operational schemes.

    Two modes:
      (A) GMM TC: provide (c_hat, params) and leave single_gaussian=None.
          - y, lam_sel from conditional_transform(x, c_hat, params)
          - theoretical: reverse water-filling Shannon RD + label cost
          - operational: uniform quantization + empirical entropy rate + label cost

      (B) Single-Gaussian TC: provide single_gaussian and set (c_hat=None, params=None).
          - y, lam_sel from single_gaussian_transform(x, sg)
          - label cost is 0 by default (no mode index)

    Returns:
        R_th: (G,) theoretical rate (bits/vector)
        D_th: (G,) theoretical distortion (MSE per dimension)
        R_op: (G,) operational rate estimate (bits/vector)
        D_op: (G,) operational distortion from reconstructed x_hat (MSE per dimension)
    """
    device = x.device
    dtype = x.dtype

    if single_gaussian is None:
        if c_hat is None or params is None:
            raise ValueError("For GMM TC, provide both c_hat and params")
        y_all, lam_sel = conditional_transform(x, c_hat, params)
        K = params.pi.numel()
        if H_label_override is None:
            H_label_th = label_entropy_from_pi(params.pi)
        else:
            H_label_th = float(H_label_override)
    else:
        # Single Gaussian baseline: no mode indices
        y_all, lam_sel = single_gaussian_transform(x, single_gaussian)
        K = 1
        H_label_th = float(H_label_override) if H_label_override is not None else 0.0

    R_th_list, D_th_list = [], []
    R_op_list, D_op_list = [], []

    for g in gammas.tolist():
        # Theoretical RD (Shannon) using lam_sel
        _, r_sum, active_mask_th = theoretical_rwf_rate(lam_sel, gamma=float(g))
        d_vec = theoretical_distortion_from_lam(lam_sel, gamma=float(g))

        R_th = r_sum.mean().item() + H_label_th
        D_th = d_vec.mean().item()

        # Operational RD (quantize + decode)
        q, delta, active_mask = tc_encode_y_given(y_all, lam_sel, gamma=float(g))
        if single_gaussian is None:
            # GMM inverse transform
            _, x_hat = tc_decode_y_oracle(q, c_hat, params, delta=delta, active_mask=active_mask, dtype=dtype)
            R_op = operational_rate_bits_per_vector(
                q, active_mask, c_hat, K=params.pi.numel(),
                H_label_override=(H_label_th if H_label_override is not None else None),
            )
        else:
            # Single Gaussian inverse transform
            y_hat = torch.zeros_like(q, dtype=dtype)
            if active_mask.any():
                y_hat[active_mask] = uniform_mid_tread_dequantize(q[active_mask], delta=delta, dtype=dtype)
            x_hat = single_gaussian_inverse_transform(y_hat, single_gaussian)

            # No label cost; entropy only over quantizer indices
            dummy_c = torch.zeros((x.shape[0],), dtype=torch.long)
            R_op = operational_rate_bits_per_vector(q, active_mask, dummy_c, K=1, H_label_override=H_label_th)
        

        mse_vec = mse_per_vector(x, x_hat)   # (N_vec,)
        D_op = mse_vec.mean().item()          # scalar MSE
        
        R_th_list.append(R_th)
        D_th_list.append(D_th)
        R_op_list.append(R_op)
        D_op_list.append(D_op)


    return (
        torch.tensor(R_th_list, device=device, dtype=dtype),
        torch.tensor(D_th_list, device=device, dtype=dtype),
        torch.tensor(R_op_list, device=device, dtype=dtype),
        torch.tensor(D_op_list, device=device, dtype=dtype),
    )


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Real GMM Transform Coding RD curves")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset generation")
    parser.add_argument("--N", type=int, default=64, help="Source dimension N")
    parser.add_argument("--K", type=int, default=16, help="Number of mixture components K")
    parser.add_argument("--use_em", action="store_true", help="Fit GMM parameters with EM and use them instead of oracle params")
    parser.add_argument("--em_iter", type=int, default=50, help="Number of EM iterations")
    parser.add_argument("--em_eps", type=float, default=1e-6, help="Diagonal jitter for covariance regularization")
    parser.add_argument("--em_T", type=int, default=20000, help="Number of training samples used for EM (<= T_train)")
    parser.add_argument("--gamma_min", type=float, default=1e-3, help="Minimum gamma for sweep")
    parser.add_argument("--gamma_max", type=float, default=10.0, help="Maximum gamma for sweep")
    parser.add_argument("--num_gamma", type=int, default=25, help="Number of gamma points")
    parser.add_argument("--batch", type=int, default=4096, help="Batch size used for RD estimation")
    parser.add_argument("--eval_perm", action="store_true", help="Report best-permutation label accuracy (handles EM label switching)")
    parser.add_argument("--save", type=str, default="", help="Optional path to save the RD plot (e.g., rd.png)")
    args = parser.parse_args()

    train_ds, test_ds, params_true = make_gmm_train_test_datasets(seed=args.seed, N=args.N, K=args.K)
    params = params_true
    print("True: ", params)

    # Optional: EM-fit parameters using training dataset (full-covariance)
    if args.use_em:
        X_em = train_ds.x[: min(args.em_T, len(train_ds))].to(torch.float32)
        print(f"[EM] fitting on X_em shape={tuple(X_em.shape)}")
        params_em = gmm_em_fit_fullcov(
            X_em,
            K=args.K,
            num_iter=args.em_iter,
            eps=args.em_eps,
            seed=args.seed,
            verbose=True,
        )
        print("[EM] learned pi:", params_em.pi.detach().cpu().numpy())
        params = params_em
    print("EM: ", params_em)

    # Rate-distortion & mode detection accuracy evaluation using test dataset
    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    print(f"[Eval split] test | dataset size={len(test_ds)} | N={args.N} K={args.K}")

    # Use one batch to estimate RD curves (increase batch for smoother curves)
    x, y_true = next(iter(loader))
    print("[DataLoader]", x.shape, x.dtype, y_true.shape, y_true.dtype)

    # MAP mode prediction under the (oracle or EM-fitted) parameters
    c_hat, _ = predict_modes(x, params_true, return_logp=False)
    
    # Raw accuracy is only meaningful when component indices align with dataset labels.
    raw_acc = (c_hat == y_true).float().mean().item()
    print(f"[MAP] raw label accuracy (may be 0 due to EM label switching): {raw_acc:.4f}")

    if args.eval_perm:
        K = params_true.pi.numel()
        cm = confusion_matrix(y_true, c_hat, K=K)
        best_acc, best_perm, c_hat_mapped = best_label_permutation(y_true, c_hat, K=K)
        print("[MAP] confusion matrix (rows=true, cols=pred):\n", cm.numpy())
        print(f"[MAP] best-permutation accuracy: {best_acc:.4f}")
        print(f"[MAP] best permutation (pred_label -> mapped_true_label): {best_perm}")

        # NOTE: We DO NOT replace c_hat used for transform coding.
        # Label permutation is only for reporting against dataset labels.

    # Gamma sweep (log-spaced)
    gammas = torch.logspace(
        math.log10(args.gamma_min),
        math.log10(args.gamma_max),
        steps=args.num_gamma,
        dtype=x.dtype,
        device=x.device,
    )

    curves = []     # list of (name, R, D)

    # (1) Theoretical (perfect GMM params + oracle pi for label cost)
    #   This corresponds to Shannon RD under reverse water-filling, plus H(pi_true).
    H_pi_true = label_entropy_from_pi(params_true.pi)
    R_th, D_th, _, _ = sweep_rd_curves(
        x,
        c_hat=c_hat,
        params=params_true,
        gammas=gammas,
        H_label_override=H_pi_true,
        single_gaussian=None,
    )
    curves.append(("Theoretical R-D bound", R_th, D_th))

    # (2) Perfect GMM params + uniform scalar quantization (operational)
    #   Use oraacle H(pi_true) for mode index cost.
    _, _, R_op, D_op = sweep_rd_curves(
        x,
        c_hat=c_hat,
        params=params_true,
        gammas=gammas,
        H_label_override=H_pi_true,
        single_gaussian=None,
    )
    curves.append(("Perfect GMM params + Uniform", R_op, D_op))

    # (3) EM-fitted GMM params + uniform scalar quanitzation (operational)
    if args.use_em:
        # params currently already replaced by EM-fit above.
        # For reporting label cost, use H(pi_hat) from EM.
        H_pi_hat = label_entropy_from_pi(params.pi)
        c_hat_em, _ = predict_modes(x, params, return_logp=False)

        _, _, R_em, D_em = sweep_rd_curves(
            x,
            c_hat=c_hat_em,
            params=params,
            gammas=gammas,
            H_label_override=H_pi_hat,
            single_gaussian=None,
        )
        curves.append(("EM - GMM params + Uniform", R_em, D_em))

    # (4) Single-mode transform coding (single Gaussian) + masking + uniform quantization
    #   Fit a single Gaussian on training dataset, evaluate RD on test split.
    sg = fit_single_gaussian_fullcov(train_ds.x[: min(args.em_T, len(train_ds))], eps=args.em_eps)
    _, _, R_sg, D_sg = sweep_rd_curves(
        x,
        c_hat=None,
        params=None,
        gammas=gammas,
        H_label_override=H_pi_true,
        single_gaussian=sg,
    )
    curves.append(("EM - single Gaussian params + Uniform", R_sg, D_sg))


    # Plot: Distortion (x-axis) vs Rate (y-axis)
    # plt.figure()
    # for name, R, D in curves:
    #     order = torch.argsort(D)
    #     Dp = D[order].cpu().numpy()
    #     Rp = R[order].cpu().numpy()
    #     plt.plot(Dp, Rp, linestyle="-")

    # plt.xlabel("Distortion (MSE per dimension)")
    # plt.ylabel("Rate (bits per vector)")
    # plt.title(f"Rate-Distortion Curves (N={args.N}, K={args.K})")
    # plt.grid(True)
    # plt.legend([c[0] for c in curves], loc="best")

    # if args.save:
    #     plt.savefig(args.save, bbox_inches="tight", dpi=200)
    #     print(f"[Saved] {args.save}")
    # else:
    #     plt.show()
    pixels_per_vector = args.N      # 8x8 grayscale면 64, RGB면 192
    MAX = 1.0                     # 정규회되어있음
    eps = 1e-12

    plt.figure()
    for name, R, D in curves:
        # bpp
        bpp = (R / pixels_per_vector).cpu().numpy()

        # PSNR
        D_np = D.cpu().numpy()
        psnr = 10 * np.log10((MAX**2) / (D_np + eps))

        order = np.argsort(bpp)
        plt.plot(bpp[order], psnr[order], linestyle="-")

    plt.xlabel("Rate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"PSNR-bpp Curves (N={args.N}, K={args.K})")
    plt.grid(True)
    plt.legend([c[0] for c in curves], loc="best")
    plt.show()
