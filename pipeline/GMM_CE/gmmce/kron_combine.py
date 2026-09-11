"""
GMM kron estimator (Sec. IV-C): combine two independently-fitted 1D GMMs
(time domain, dim Nt, Kt components; frequency domain, dim Nc, Kc
components) into a single joint K=Kt*Kc-component GMM via Kronecker
products, then refit the mixing coefficients with a single E-step over
the full (unvectorized-pilot) training data, exactly as described in the
paper.
"""
from __future__ import annotations

import numpy as np

from .complex_gmm import GMM, _log_gauss_full


def combine_kronecker(gmm_time: GMM, gmm_freq: GMM, X_train: np.ndarray, reg: float = 1e-6) -> GMM: # 논문에서 IV-C. Kronecker Estimator에 해당하는 함수. full은 D = Nc*Nt의 긴 길이로 했지만 여기선 D = Nt, D = Nc 따로 작게 학습하고 Kronecker 곱으로 표현.
    """X_train: (N, Nc*Nt) full (pilot-free) training vectors, vec()-ordered
    (time outer/slow, freq inner/fast) -- used only to refine the mixing
    coefficients via one E-step."""
    Kt, Kc = gmm_time.K, gmm_freq.K
    K = Kt * Kc
    D = gmm_time.D * gmm_freq.D   # Nt * Nc

    means = np.empty((K, D), dtype=np.complex128)
    covs = np.empty((K, D, D), dtype=np.complex128)
    init_w = np.empty(K)
    idx = 0
    for kt in range(Kt):
        for kc in range(Kc): # 모든 Kt kC 조합에 대해 다 함.
            means[idx] = np.kron(gmm_time.means[kt], gmm_freq.means[kc])  # 시간, 주파수 독립 가정
            covs[idx] = np.kron(gmm_time.covs[kt], gmm_freq.covs[kc])     # 시간, 주파수 독립 가정
            init_w[idx] = gmm_time.weights[kt] * gmm_freq.weights[kc]     # 시간, 주파수 독립 가정
            idx += 1

    N = X_train.shape[0]
    log_lik = np.empty((K, N))
    for k in range(K):
        log_lik[k] = np.log(init_w[k] + 1e-300) + _log_gauss_full(X_train, means[k], covs[k], reg)
    m = log_lik.max(axis=0, keepdims=True)
    resp = np.exp(log_lik - m)
    resp /= resp.sum(axis=0, keepdims=True)           # E-step 딱 한번만 돌림.
    weights = resp.mean(axis=1)
    weights /= weights.sum()

    return GMM(weights, means, covs)
