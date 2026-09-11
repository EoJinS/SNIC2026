"""
Genie power-delay-profile (PDP) / Doppler-spectrum (DS) baseline
estimators, Sec. III of the paper. These are "utopian" LMMSE baselines
that use the *true* instantaneous channel of the very sample being
estimated to compute the PDP/DS (hence "genie"): a best-possible
performance bound for methods that only use PDP/DS side information
(no learned prior).
"""
from __future__ import annotations

import numpy as np

from .structured import dft_matrix
from .pilots import PilotGrid


def genie_pdp_ds(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]: # 지니 channel을 입력받아 D-D domain으로 변환해 (N, Nc)사이즈 pdp랑 (N, Nt) 사이즈 ds 반환.
    """H: (N, Nc, Nt). Returns (pdp, ds):
       pdp: (N, Nc) delay-domain power, averaged over the Nt time symbols
       ds:  (N, Nt) Doppler-domain power, averaged over the Nc carriers
    """
    N, Nc, Nt = H.shape # H는 3차원임! sample by Nc by Nt
    F_Nc = dft_matrix(Nc)
    F_Nt = dft_matrix(Nt)

    # p_i = |F_Nc^H h_i|^2 for each time-symbol column h_i, averaged over i
    delay_domain = np.einsum('dc,nct->ndt', F_Nc.conj().T, H)  # (N, Nc, Nt) 걍 H의 2번째 축인 주파수 영역에 IDFT해서 delay도매인 만듬. 코드 설명 자세히 말하면 첫 번째 재료는 축 이름이 (d,c), 두 번째 재료는 축 이름이 (n,c,t)인데, 공통으로 등장하는 c는 곱하고 더해서 없애고, n,d,t는 그대로 남겨라
    pdp = np.mean(np.abs(delay_domain) ** 2, axis=2)  # (N, Nc) 

    # d_k = |F_Nt g_k|^2 for each carrier row g_k, averaged over k
    doppler_domain = np.einsum('dt,nct->ncd', F_Nt, H)  # (N, Nc, Nt) 걍 H의 1번째 축인 시간 영역에 DFT해서 doppler도매인 만듬. 코드 설명 자세히 말하면 첫 번째 재료는 축 이름이 (d,t), 두 번째 재료는 축 이름이 (n,c,t)인데, 공통으로 등장하는 t는 곱하고 더해서 없애고, n,c,d는 그대로 남겨라
    ds = np.mean(np.abs(doppler_domain) ** 2, axis=1)  # (N, Nt)
    return pdp, ds


def covariances_from_pdp_ds(pdp: np.ndarray, ds: np.ndarray, Nc: int, Nt: int): # pdp랑 ds로 Covariance matric를 만드는 함수.
    """pdp: (N, Nc), ds: (N, Nt). Returns (C_freq, C_time), each (N, D, D)."""
    F_Nc = dft_matrix(Nc)
    F_Nt = dft_matrix(Nt)
   
    C_freq = np.einsum('cf,nf,gf->ncg', F_Nc, pdp, F_Nc.conj(), optimize=True) # (N, Nc, Nc) >> C_freq = F_Nc diag(p) F_Nc^H Section III 근거로.
    C_time = np.einsum('fc,nf,fg->ncg', F_Nt.conj(), ds, F_Nt, optimize=True) # (N, Nt, Nt) >> C_time = F_Nt^H diag(d) F_Nt Section III 근거로.

    return C_freq, C_time


def pdp_ds_kron_estimate(H_test_true: np.ndarray, grid: PilotGrid, sigma2: float, # 실제 정답채널 H에서 pdp랑 ds만을 뽑아서 알고 있는 상태에서, 이를 토대로 채널 추정 H^하는 함수. GMM없이 채널추정하는 baseline기법.
                          rng: np.random.Generator) -> np.ndarray: # 정답 채널을 가져다 써서 pdp, ds를 뽑아쓴다는 점에서 현실에선 불가능한 이상적인 기법임.
    """Genie PDP/DS estimator with a full Kronecker covariance
    C = kron(C_time, C_freq), zero-mean LMMSE, Sec. III + IV-C combo."""
    N, Nc, Nt = H_test_true.shape
    pdp, ds = genie_pdp_ds(H_test_true)
    C_freq, C_time = covariances_from_pdp_ds(pdp, ds, Nc, Nt)  # (N,Nc,Nc), (N,Nt,Nt)


    # pdp랑 ds는 통계적 성질일 뿐이지 실제 체널값은 아님. 그래서 파일럿 부분으로 실제 체널 모양을 유추해야함.  
    Ct_p = C_time[:, grid.pilot_symbols][:, :, grid.pilot_symbols]   # (N, Npt, Npt) 시간 공분산중에 파일럿부분만 뽑기
    Cf_p = C_freq[:, grid.pilot_carriers][:, :, grid.pilot_carriers]  # (N, Npc, Npc) 주파수 공분산중에 파일럿부분만 뽑기

    true_pilots = H_test_true[:, grid.pilot_carriers][:, :, grid.pilot_symbols]  # (N,Npc,Npt) 파일럿 위치쪽에 실제 채널 H값. 
    noise = (rng.normal(size=true_pilots.shape) + 1j * rng.normal(size=true_pilots.shape)) * np.sqrt(sigma2 / 2)
    y_grid = true_pilots + noise

    H_hat = np.empty((N, Nc, Nt), dtype=np.complex128)
    for n in range(N):
        Cy = np.kron(Ct_p[n], Cf_p[n]) + sigma2 * np.eye(grid.Np)  # 
        y = y_grid[n].T.reshape(-1)                                # 샘플의 노이즈 낀 값을 한줄로 펴기
        sol = np.linalg.solve(Cy, y)
        Ct_cols = C_time[n][:, grid.pilot_symbols]   # (Nt, Npt)
        Cf_cols = C_freq[n][:, grid.pilot_carriers]  # (Nc, Npc)
        CAh = np.kron(Ct_cols, Cf_cols)             
        h_hat = CAh @ sol                            # LMMSE: h^= ChA^H(AChA^H+Cn)^(-1)y
        H_hat[n] = h_hat.reshape(Nt, Nc).T
    return H_hat


def pdp_ds_2x1d_estimate(H_test_true: np.ndarray, grid: PilotGrid, sigma2: float,  # GMM 2×1D의 genie 버전. 매 샘플의 진짜 PDP/DS로 만든 공분산을 쓰되, 한 방에 D차원을 다 푸는 대신 주파수→시간 순서로 두 단계에 걸쳐 저차원으로 나눠서 푸는 방식.  
                          rng: np.random.Generator, Cn_eff: float | None = None) -> tuple[np.ndarray, float]: # 중요!! 여기서도 실제 정답 H와 1차 주파수방향 LMMSE를 사용해 얻은 값과 차이를 eff.noise라 명명. 근데 이것 또한 지니라 가능한거임. 실제상황에선 이걸 몰라.
    """Genie PDP/DS estimator with cascaded 1D LMMSE (frequency then time),
    mirroring GMM 2x1D. If Cn_eff is None it is calibrated internally from
    this same batch (since PDP/DS covariances are per-sample/genie, there
    is no separate offline "training" phase to calibrate from)."""
    N, Nc, Nt = H_test_true.shape
    pdp, ds = genie_pdp_ds(H_test_true)
    C_freq, C_time = covariances_from_pdp_ds(pdp, ds, Nc, Nt)

    Cf_p = C_freq[:, grid.pilot_carriers][:, :, grid.pilot_carriers]  # (N,Npc,Npc) C_freq에서 pilot 있는부분만 뽑기,  A C_freq A^H
    true_pilots = H_test_true[:, grid.pilot_carriers][:, :, grid.pilot_symbols]  # (N,Npc,Npt) 파일럿 위치쪽에 실제 채널 H값. 
    noise = (rng.normal(size=true_pilots.shape) + 1j * rng.normal(size=true_pilots.shape)) * np.sqrt(sigma2 / 2)
    y_grid = true_pilots + noise  # (N, Npc, Npt)

    # Stage 1: per pilot time-symbol, 주파수방향으로 LMMSE using C_freq[n]: 원래 C(Nc*Nt by Nc*Nt)보다 작은 C_freq(Nc by Nc)로 연산
    H_stage1 = np.empty((N, Nc, grid.Npt), dtype=np.complex128)
    Cy1 = Cf_p + sigma2 * np.eye(grid.Npc)[None]
    sol1 = np.linalg.solve(Cy1, y_grid)  # (N, Npc, Npt)   
    Cf_cols = C_freq[:, :, grid.pilot_carriers]  # (N, Nc, Npc)   CA^H
    H_stage1 = np.einsum('nck,nkj->ncj', Cf_cols, sol1)  # (N, Nc, Npt) h^ = C_freq A^H(A C_freq A^H+Cn)^(-1)y. 

    if Cn_eff is None: # 중요!! 여기서도 실제 정답 H와 1차 주파수방향 LMMSE를 사용해 얻은 값과 차이를 eff.noise라 명명. 근데 이것 또한 지니라 가능한거임. 실제상황에선 이걸 몰라.
        true_at_pilot_symbols = H_test_true[:, :, grid.pilot_symbols]
        Cn_eff = float(np.mean(np.abs(H_stage1 - true_at_pilot_symbols) ** 2))

    # Stage 2: per carrier, time-domain 시간방향으로 LMMSE using C_time[n]: 원래 C(Nc*Nt by Nc*Nt)보다 작은 C_time(Nt by Nt)로 연산
    Ct_p = C_time[:, grid.pilot_symbols][:, :, grid.pilot_symbols]  # (N,Npt,Npt)     A C_time A^H     
    Cy2 = Ct_p + Cn_eff * np.eye(grid.Npt)[None]                    # A C_time A^H + Cn  
    sol2 = np.linalg.solve(Cy2, np.transpose(H_stage1, (0, 2, 1)))  # (N, Npt, Nc)  (A C_time A^H + Cn.eff)^(-1)    C_freq A^H(A C_freq A^H+Cn)^(-1)y
    Ct_cols = C_time[:, :, grid.pilot_symbols]  # (N, Nt, Npt)        >> C_time A^H (A C_time A^H + Cn.eff)^(-1) C_freq A^H(A C_freq A^H+Cn)^(-1)y
    H_hat = np.einsum('ntj,njc->nct', Ct_cols, sol2)  # (N, Nc, Nt)
    return H_hat, Cn_eff
