"""
Complex-valued (circularly-symmetric) Gaussian-mixture EM fitting.

Three flavours are implemented, all sharing the same E-step:

  * `fit_full_gmm`       -- unconstrained covariances (GMM full)
  * `fit_diagonal_gmm`   -- diagonal covariances, used after transforming
                            data into the DFT domain (GMM b-circ / 2x1D-circ)
  * `fit_toeplitz_gmm`   -- (block-)Toeplitz-constrained covariances via
                            the circulant-embedding EM update of
                            Barton & Fuhrmann (1993), Sec. IV-A of the
                            paper (GMM b-toep / 2x1D-toep)

sklearn's GaussianMixture only supports real-valued data, so these are
implemented from scratch with numpy/scipy, vectorized over the K
components and the N samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .structured import block_operator


@dataclass
class GMM:
    """A fitted complex GMM: K components of dimension D."""
    weights: np.ndarray   # (K,)
    means: np.ndarray     # (K, D) complex
    covs: np.ndarray      # (K, D, D) complex

    @property
    def K(self) -> int:
        return len(self.weights)

    @property
    def D(self) -> int:
        return self.means.shape[1]  # shape이 (K, D)이니


def _kmeanspp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray: # EM 시작하기 전 K개의 "좋은" 초기 평균값(논문에서 mu_k)을 고름.좋다는 의미는 K개의 mean값이 충분히 퍼트려지게 초기값 세팅해야 EM 알고리즘이 잘 수렴.
    """k-means++-style seeding for complex data (distance = |.|^2)."""
    N = X.shape[0]
    idx = [rng.integers(N)]                          # 아무 점이나 하나 랜덤으로 첫 초기값
    d2 = np.sum(np.abs(X - X[idx[0]]) ** 2, axis=1)  # 그 점 기준으로 X 전체 점까지 거리^2 (** 2가 제곱) 배열, X는 (N,D)임. (샘플수) by (Nc*Nt). np.sum(  ,axis=1)이므로 샘플별로 거리합이 들어간 배열임.
    for _ in range(1, K):                            # for _ 는 그냥 반복만 한다는 의미.
        probs = d2 / d2.sum()                        # 가장 확률 높은(기존 샘플과 거리가 가장 먼) 샘플을 높은 확률로 선택하게 함.
        i = rng.choice(N, p=probs)                   # 확률도 벡터임.
        idx.append(i)                                # 파이썬리스트(넘파이 배열은 안됨)를 이어 붙이는 용도.
        d2 = np.minimum(d2, np.sum(np.abs(X - X[i]) ** 2, axis=1)) # 그 점 기준으로 새로운 X 전체 점까지 거리^2 업데이트 
    return X[np.array(idx)].copy()                   # N by D의 X중에서 통계적 거리(2 norm)가 충분히 멀리 떨어진 샘플을 idx개수(K)만큼 추려  (K) by (Nc*Nt)의 matrix 반환.(나중에 메모리주소 타고와 수정되는걸 방지하기 위해 안전빵으로 copy본 전달)


def _log_gauss_full(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, reg: float) -> np.ndarray: # 이 데이터가 k번째 가우시안에서 나왔을 가능성이 얼마나 되는지 계산할떄 써먹음. log는 컴퓨터에서 안정적 계산을 위해 적용.
    """log CN(x; mean, cov) for all rows of X (N, D). Returns (N,)."""
    D = X.shape[1]
    cov_reg = cov + reg * np.eye(D) # reg 라는 매우 작은 값을 더해 Positive Definite 행렬로 만듬.
    c, low = cho_factor(cov_reg, lower=True) # Positive Definite이라 숄레스키 분해 가능. c는 하삼각행렬 저장하고, low는 하삼각이면 true 반환.
    logdet = 2.0 * np.sum(np.log(np.abs(np.diag(c))))
    Xc = (X - mean).T  # (D, N)
    sol = cho_solve((c, low), Xc)  # C 역행렬 구하는데 숄레스키 활용해 빠르게 
    maha = np.real(np.sum(Xc.conj() * sol, axis=0))  
    return -D * np.log(np.pi) - logdet - maha  # 길이 N짜리 배열의 complex gaussian 확률밀도의 로그값을 반환.


def _e_step_full(X: np.ndarray, gmm: GMM, reg: float) -> tuple[np.ndarray, float]:
    N, K = X.shape[0], gmm.K
    log_resp = np.empty((N, K)) # 샘플별로 k번째 component일 확률 계산해 넣기 위한 빈 공간 (N, K)
    for k in range(K):
        log_resp[:, k] = np.log(gmm.weights[k] + 1e-300) + _log_gauss_full(X, gmm.means[k], gmm.covs[k], reg) # k번째 컴포넌트 고를 확률 * k조건부 확률분포.
    m = log_resp.max(axis=1, keepdims=True)                                                # 각 행, sample별로 가장 큰 k값(log k)을 저장함. 샘플수 길이에 해당하는 벡터 나옴.
    log_norm = m + np.log(np.sum(np.exp(log_resp - m), axis=1, keepdims=True) + 1e-300)    # log p(x_n) = log sum_k p(k)CN(x_n;mu_k,C_k), (N,1)
    log_resp -= log_norm                                                                    # log_resp를, exp() 했을 때 각 행(샘플)의 합이 정확히 1이 되도록 만듬. 정규화
    total_loglik = float(np.sum(log_norm))                                                  # EM이 실제로 최적화하는 관측데이터 로그우도 (Q함수가 아니라 진짜 log-likelihood)
    return np.exp(log_resp), total_loglik                                                   # 샘플별로 k번째 component일 확률인 (N, K) 행렬 + 전체 로그우도 반환.


def fit_full_gmm(X: np.ndarray, K: int, n_iter: int = 25, reg: float = 1e-6, # X(학습 데이터), K(컴포넌트 개수), n_iter(EM 최대 반복 횟수, 기본 25), reg(안전장치 크기)
                  seed: int | None = None, verbose: bool = False, tol: float = 1e-4) -> GMM:  # tol(로그우도 상대증가량이 이 밑으로 떨어지면 조기 종료)
    """Unconstrained-covariance complex GMM EM. X: (N, D) complex."""
    N, D = X.shape
    rng = np.random.default_rng(seed)
    means = _kmeanspp_init(X, K, rng)                                        # mu_k 생성(k별로 쌓아 (K) by (Nc*Nt)의 matrix 임.)
    global_cov = np.cov(X.T) if D > 1 else np.array([[np.var(X)]])           # 입력된 data로 (Nc*Nt) by (Nc*Nt), D by D의 covariance matrix를 구하려고
    covs = np.tile((global_cov + reg * np.eye(D))[None], (K, 1, 1))          # np.tile은 그냥 쌓아올리는 명령어. (K, 1, 1)이니 첫번째 axis만 k번 복사해 쌓고 두,세번째는 그대로. >> k개의 covariance를 동일하게 초기화한 상태.
    weights = np.full(K, 1.0 / K)                                            # 일단은 unif으로 세팅.
    gmm = GMM(weights, means, covs)                                          # 초기값으로 세팅된 gmm k개 생성.

    prev_ll = -np.inf                                                         # EM이 실제로 최적화하는 로그우도를 매 iteration 추적 (수렴 판단용)
    for it in range(n_iter):
        resp, ll = _e_step_full(X, gmm, reg)                                  # E step: (N, K) 책임값 + 이번 iteration의 전체 로그우도
        Nk = resp.sum(axis=0) + 1e-12                                         # (K,) 이후부터 M step.
        weights = Nk / N                                                      # k 일 확률
        means = (resp.T @ X) / Nk[:, None]                                    # M-step에서 가중치 매겨진 로그가능도를 미분해 0이 되는, 이를 최대로 하는 평균값을 구하면 저렇게 수식정리 가능.
        covs = np.empty((K, D, D), dtype=np.complex128)                       # M-step에서 가중치 매겨진 로그가능도를 미분해 0이 되는, 이를 최대로 하는 Cov를 구하면 저렇게 수식정리 가능.
        for k in range(K):
            Xc = X - means[k]
            w = resp[:, k]
            covs[k] = (Xc * w[:, None]).T @ Xc.conj() / Nk[k]
            covs[k] += reg * np.eye(D)
        gmm = GMM(weights, means, covs)
        if verbose:
            print(f"[full-gmm] iter {it}: avg log-lik = {ll / N:.4f}")
        if abs(ll - prev_ll) < tol * abs(prev_ll) + 1e-8:                     # 로그우도가 더 이상 유의미하게 안 느는지(EM의 진짜 수렴 기준) 체크
            break
        prev_ll = ll
    return gmm


# ---------------------------------------------------------------------
# Diagonal-covariance EM (for circulant-constrained variants: fit in the
# DFT-transformed domain where the target covariance is diagonal).
# ---------------------------------------------------------------------

def fit_diagonal_gmm(X: np.ndarray, K: int, n_iter: int = 25, reg: float = 1e-6,
                      seed: int | None = None, verbose: bool = False, tol: float = 1e-4) -> GMM:
    """Diagonal-covariance complex GMM EM. Returns a GMM whose `covs` are
    diagonal matrices (still stored densely as (K, D, D) for a uniform
    interface, but cheap to fit: O(D) per sample per component)."""
    N, D = X.shape
    rng = np.random.default_rng(seed)
    means = _kmeanspp_init(X, K, rng)
    var0 = np.var(X, axis=0).real + reg      # D 길이 벡터.  N by D 인 X데이터를 N축이 없어지게 var 계산. 즉 D별로 var 계산됨.
    variances = np.tile(var0[None], (K, 1))  # (K, D) k마다 애초에 variance의 대각성분만 만들어둠. full의 경우는 (K, D, D)였음.
    weights = np.full(K, 1.0 / K)

    prev_ll = -np.inf
    for it in range(n_iter):
        log_resp = np.empty((N, K))
        for k in range(K):
            v = variances[k] + reg    # v는 대각성분 variance하나
            Xc = X - means[k]
            log_resp[:, k] = (np.log(weights[k] + 1e-300)
                               - np.sum(np.log(np.pi * v))
                               - np.sum(np.abs(Xc) ** 2 / v, axis=1))
        m = log_resp.max(axis=1, keepdims=True)
        log_norm = m + np.log(np.sum(np.exp(log_resp - m), axis=1, keepdims=True) + 1e-300)  # log p(x_n)
        resp = np.exp(log_resp - log_norm)                                                    # 여기까지 E step. diag 꼴이라 detC, inv(C)가 매우 단순해져 간단한 form으로 작성.
        ll = float(np.sum(log_norm))                                                          # 전체 로그우도

        Nk = resp.sum(axis=0) + 1e-12
        weights = Nk / N
        means = (resp.T @ X) / Nk[:, None]
        for k in range(K):
            Xc = X - means[k]
            variances[k] = (resp[:, k] @ np.abs(Xc) ** 2) / Nk[k] + reg                                     # 여기까지 M step. Cov 계산과정도 매우 단순해짐.
        if verbose:
            print(f"[diagonal-gmm] iter {it}: avg log-lik = {ll / N:.4f}")
        if abs(ll - prev_ll) < tol * abs(prev_ll) + 1e-8:                                                    # 로그우도 수렴 체크
            break
        prev_ll = ll

    covs = np.zeros((K, D, D), dtype=np.complex128)
    for k in range(K):
        covs[k] = np.diag(variances[k].astype(np.complex128))
    return GMM(weights, means, covs)


def fit_circulant_gmm(X: np.ndarray, K: int, dims: list[int], n_iter: int = 25,  # "circulant 공분산은 DFT 도메인에서 정확히 대각행렬이 된다는 성질을 이용해서, 데이터 자체를 DFT 도메인으로 변환한다. 그럼 그 데이터의 공분산은 반드시 대각화되므로, 그 변환된 데이터에 fit_diagonal_gmm으로 EM을 새로 돌리고, 결과(평균·공분산)만 다시 원래 도메인으로 역변환하는 함수"
                       reg: float = 1e-6, seed: int | None = None, tol: float = 1e-4) -> GMM:
    """(Block-)circulant-constrained complex GMM (Sec. IV-B): fit a
    diagonal-covariance GMM in the (block-)DFT-transformed domain, then
    map back to the original domain (GMM b-circ / 2x1D-circ).

    `dims` in outer->inner order, e.g. [Nt, Nc] for the 2D case or [Nc]
    (resp. [Nt]) for the 1D sub-estimators used by the 2x1D cascade.
    """
    F = block_operator(dims, kind='dft')  # unitary (D, D)
    Fh = F.conj().T
    X_tilde = X @ F.T  # each row: x_tilde = F @ x  (row form: x_tilde^T = x^T @ F^T)
    gmm_tilde = fit_diagonal_gmm(X_tilde, K, n_iter=n_iter, reg=reg, seed=seed, tol=tol)

    # mu_k = F^H mu_tilde_k  (column form); row form: mu_tilde_k^T @ (F^H)^T
    means = gmm_tilde.means @ Fh.T
    # C_k = F^H diag(c_k) F. gmm_tilde.covs is diagonal, so pull the
    # diagonal and form each C_k as (Fh * d_k) @ F -- O(K D^2) space,
    # O(K D^3) time. (The dense einsum 'ij,kjl,lm->kim' has a naive
    # O(K D^4) contraction path and blows up past D~256.)
    d = np.real(np.diagonal(gmm_tilde.covs, axis1=1, axis2=2))  # (K, D)
    D = F.shape[0]
    covs = np.empty((K, D, D), dtype=np.complex128)
    for k in range(K):
        covs[k] = (Fh * d[k][None, :]) @ F
    return GMM(gmm_tilde.weights, means, covs)


# ---------------------------------------------------------------------
# (Block-)Toeplitz-constrained EM (Barton & Fuhrmann 1993 projection,
# paper Eqs. (5)-(7)).
# ---------------------------------------------------------------------

def fit_toeplitz_gmm(X: np.ndarray, K: int, dims: list[int], n_iter: int = 25,
                      reg: float = 1e-6, seed: int | None = None,
                      verbose: bool = False, tol: float = 1e-4) -> GMM:
    """Toeplitz(-block)-constrained complex GMM EM.

    `dims` gives the block structure in outer->inner order, e.g. [Nt, Nc]
    for the full 2D (block-Toeplitz-with-Toeplitz-blocks) case, or e.g.
    [Nc] for a plain 1D Toeplitz covariance (used by the 2x1D-toep
    sub-estimators).
    """
    N, D = X.shape
    Qt = block_operator(dims, kind='toeplitz')  # (4D, D) with orthonormal columns 본논문에서 Q~임.
    fourD = Qt.shape[0]                          # 4D
    rng = np.random.default_rng(seed)

    means = _kmeanspp_init(X, K, rng)
    weights = np.full(K, 1.0 / K)

    # Initialize c_k as a flat (white) spectrum whose total power matches
    # the global sample covariance; Qt^H Qt = I_D so this gives an
    # isotropic starting covariance (trace/D) * I_D, refined by EM below.
    global_cov = np.cov(X.T) if D > 1 else np.array([[np.var(X)]], dtype=np.complex128)
    c0_val = max(np.real(np.trace(global_cov)) / D, reg)            # 평균적인 variance값을 초기값으로 사용할 것임.
    c = np.full((K, fourD), c0_val)                                  # k별로 c벡터들(4D길이) 모아둔 것들을 등방성 띄게 초기값 세팅.

    def cov_from_c(c_k):                                            # Toeplitz covariance를 사용하기 때문에 covariance 벡터인 c_k만 업데이트함. 그래서 벡터 c_k를 다시 covariance matrix로 만드는 코드임.
        return (Qt.conj().T * c_k[None, :]) @ Qt                    # Qt^H diag(c) Qt, (D, D) 논문에서 4번식. c_k는 4*D길이임.

    covs = np.array([cov_from_c(c[k]) + reg * np.eye(D) for k in range(K)]) # k마다 c벡터(4D길이)로 covariance matrix 형성하기.
    gmm = GMM(weights, means, covs)

    prev_ll = -np.inf
    for it in range(n_iter):
        resp, ll = _e_step_full(X, gmm, reg)                                                      # ← E-step (책임값 + 전체 로그우도)
        Nk = resp.sum(axis=0) + 1e-12                                                             # M-step ↓↓↓
        weights = Nk / N
        means = (resp.T @ X) / Nk[:, None]                               

        new_c = np.empty_like(c)     # k by 4D 빈공간 생성                                     
        new_covs = np.empty((K, D, D), dtype=np.complex128) # k by D by D 빈공간 생성

        for k in range(K):
            Xc = X - means[k]                                        # M step으로 C업데이트 하는 과정. 
            w = resp[:, k]                                           # M step으로 C업데이트 하는 과정. 
            C_hat = (Xc * w[:, None]).T @ Xc.conj() / Nk[k]          # M step으로 C업데이트 하는 과정. Q를 C로 미분해 0이 되는 C. 이때 C는 Q를 최대화할 뿐이지 toeplize란 보장 없음.

            C_k = covs[k]                                    
            Cinv = np.linalg.inv(C_k)                                # toeplize행렬은 4번식으로 대각화해도 Q~가 unitary가 아니기에 1/c대입하는 간단한 방식 불가능하고, 직접 역함수 구해야함.
            Theta = Qt @ (Cinv @ C_hat @ Cinv - Cinv) @ Qt.conj().T  # 논문에서 7번식, (2D, 2D)
            step = c[k] * np.real(np.diag(Theta)) * c[k]             # 논문에서 6번식
            c_new = c[k] + step                                      # 논문에서 6번식
            c_new = np.clip(c_new, reg, None)                        # np.clip(x, 최소값, 최대값) 즉 0안되게 방지.
            new_c[k] = c_new                                         # c^(i+1)업뎃
            new_covs[k] = cov_from_c(c_new) + reg * np.eye(D)        # C^(i+1)업뎃

        c = new_c
        covs = new_covs
        gmm = GMM(weights, means, covs)                                                               # ← M-step 결과로 gmm 갱신
        if verbose:
            print(f"[toeplitz-gmm] iter {it}: avg log-lik = {ll / N:.4f}")
        if abs(ll - prev_ll) < tol * abs(prev_ll) + 1e-8:                                              # 로그우도 수렴 체크
            break
        prev_ll = ll

    return gmm
