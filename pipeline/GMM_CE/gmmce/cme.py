"""
The GMM-based conditional mean estimator (CME), Eqs. (2)-(3) of the paper,
applied jointly (GMM full / b-toep / b-circ / kron) and in cascaded 1D
form (GMM 2x1D and its variants, plus the genie PDP/DS baselines).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve # Cholesky 분해를 이용하겠다.

from .complex_gmm import GMM
from .pilots import PilotGrid, unvec


def _responsibilities_and_filters(gmm: GMM, A: np.ndarray, Cn: np.ndarray, Y: np.ndarray, reg: float = 1e-9, need_filters: bool = True): # 논문 2번 3번식에 해당. EM으로 학습된 GMM 가져와서 LMMSE로 체널 h^(K)를 추정하는 함수. 2번 수식의 p(k|y)랑 그 옆에 곱해진 값을 반환. Y는 테스트(추정) 단계에서 실제로 관측한, 노이즈 낀 파일럿 값
    """Compute p(k|y) for all test samples and the per-component LMMSE
    filtered estimates, vectorized over samples.

    Y: (Np, N_test) observations (columns = samples).
    Returns log_resp (K, N_test) [not yet normalized... actually normalized]
    and est_k (K, D, N_test) per-component filtered estimates
    (mu_k + C_k A^H Cy_k^{-1} (y - A mu_k)), or None for est_k if
    need_filters=False (skips that O(K*D*N_test) allocation entirely --
    used by responsibilities_only, which never looks at est_k).
    """
    K, D = gmm.K, gmm.D                               # GMM parameter의 dimension ex) cov는 K by D by D
    Np, N_test = Y.shape                              # Y는 열마다 샘플임.
    log_w = np.log(gmm.weights + 1e-300)

    log_lik = np.empty((K, N_test))
    est_k = np.empty((K, D, N_test), dtype=np.complex128) if need_filters else None

    for k in range(K):
        mu = gmm.means[k]
        C = gmm.covs[k]
        Amu = A @ mu  # (Np,)
        Cy = A @ C @ A.conj().T + Cn    +   reg * np.eye(Np)
        c, low = cho_factor(Cy, lower=True)
        logdet = 2.0 * np.sum(np.log(np.abs(np.diag(c))))

        diff = Y - Amu[:, None]  # (Np, N_test) 2번식에서 y-Amu_k
        sol = cho_solve((c, low), diff)  # Cy^{-1} diff, (Np, N_test)
        maha = np.real(np.sum(diff.conj() * sol, axis=0))  # (N_test,)
        log_lik[k] = log_w[k] - Np * np.log(np.pi) - logdet - maha  # 3번식에서 분자. p(k)*NC(y;Amu_k,Cy,k)에 log씌운 값.
                                                       # (D, Np), 2번식에서 filter인 Ck A^H C_y,k^(-1)임
        if need_filters:
            est_k[k] = mu[:, None] + C @ A.conj().T @ sol  # 2번식에서 p(k|y)제외한 전체. (D, N_test)

    m = log_lik.max(axis=0, keepdims=True)# 걍 np.exp(log_lik)에 너무 작은 값이 들어가면 0으로 뭉개질 수 있으니 안전장치.
    resp = np.exp(log_lik - m) # 걍 np.exp(log_lik)에 너무 작은 값이 들어가면 0으로 뭉개질 수 있으니 안전장치.

    resp /= resp.sum(axis=0, keepdims=True) # resp가 조건부 확률이 될 수 있게 K끼리 다 더한 값을 1로 정규화. 이게 2번식의 p(k|y)
    return resp, est_k


def cme_estimate(gmm: GMM, A: np.ndarray, Cn: np.ndarray, Y: np.ndarray, reg: float = 1e-9,
                  batch_size: int = 2000) -> np.ndarray:
    """Full joint GMM-CME, Eq. (2). Y: (Np, N_test). Returns h_hat (D, N_test).

    Processes N_test in batches of `batch_size` columns: the per-component
    filtered-estimate tensor est_k is O(K*D*batch_size) rather than
    O(K*D*N_test), which otherwise blows up at paper scale (K=128, D=336,
    N_test=10000 needs ~6.4GB for est_k alone in one shot -- observed to
    OOM when several K=128 GMMs are resident at once). Each test sample is
    independent (Y's columns never interact), so this is mathematically
    identical to the unbatched computation, just bounded peak memory.
    """
    D = gmm.D
    N_test = Y.shape[1]
    h_hat = np.empty((D, N_test), dtype=np.complex128)
    for start in range(0, N_test, batch_size):
        end = min(start + batch_size, N_test)
        resp, est_k = _responsibilities_and_filters(gmm, A, Cn, Y[:, start:end], reg)
        h_hat[:, start:end] = np.sum(resp[:, None, :] * est_k, axis=0) # 시그마 k에 대해 다 더하기.
    return h_hat


def responsibilities_only(gmm: GMM, A: np.ndarray, Cn: np.ndarray, Y: np.ndarray, reg: float = 1e-9) -> np.ndarray:
    """Return just p(k|y), shape (K, N_test) -- used for Fig. 2."""
    resp, _ = _responsibilities_and_filters(gmm, A, Cn, Y, reg, need_filters=False)
    return resp


def mean_components_for_responsibility(resp: np.ndarray, threshold: float = 0.99) -> float: # K개 컴포넌트 중, 상위 몇 개만 더해도 확률의 99%를 커버할 수 있는지"를 평균내서 알려주는 함수. p(k|y)를 k별로 다 더하면 1임! 그걸 이용해서 k 더해서 99% 채울때까지 필요한 components수가 몇인지 계산.
    """Average, over samples (columns of resp), of the minimal number of
    (sorted-descending) components whose cumulative responsibility
    reaches `threshold`."""
    sorted_resp = -np.sort(-resp, axis=0)  # (K, N_test), descending per column
    cum = np.cumsum(sorted_resp, axis=0)
    reached = cum >= threshold
    # first index (0-based) where reached is True, +1 for count
    counts = np.argmax(reached, axis=0) + 1
    return float(np.mean(counts)) # 샘플들 별로 몇개의 components팔요한지 계산해서 평균 쳐버리기.
