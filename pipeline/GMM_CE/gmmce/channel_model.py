"""
Synthetic doubly-selective OFDM channel generator.

The original paper (Fesl et al., "Channel Estimation based on Gaussian
Mixture Models with Structured Covariances", Asilomar 2022) generates its
training/test data with the QuaDRiGa ray-tracing-based channel simulator
(a proprietary/MATLAB tool, 3GPP 38.901 UMa scenario). QuaDRiGa is not
available in this Python environment, so we substitute a statistically
matched geometry-based stochastic channel model (GSCM) that reproduces the
same *structural* properties QuaDRiGa channels have and that the paper's
algorithms actually exploit:

  * frequency selectivity governed by a (random, per-drop) power-delay
    profile (PDP), related to the frequency covariance via the
    Wiener-Khinchin / Fourier relation (exactly the relation used in
    Sec. III of the paper),
  * time selectivity governed by a classic Jakes/Clarke Doppler spectrum
    with a per-drop mobile velocity, related to the time covariance the
    same way,
  * "environment" diversity across drops (different multipath geometries),
    which is exactly the kind of prior structure the GMM is trained to
    capture.

Each channel "drop" corresponds to one random MT position/geometry: L
random multipath components with random delays (exponential PDP with a
log-normally distributed RMS delay spread, as in 3GPP 38.901 UMa) and
random angles (giving a random per-path Doppler shift for a given MT
velocity, i.e. the classic Jakes model). This is not QuaDRiGa, but it is a
standard, textbook GSCM and preserves everything the estimators in this
repository actually use.
"""
from __future__ import annotations

from dataclasses import dataclass # 실험에서 계속 쓰는 파라미터들 편하게 고정시킬라고.

import numpy as np

SPEED_OF_LIGHT = 3e8


@dataclass # 실험에서 계속 쓰는 파라미터들 편하게 고정시킬라고.
class SystemConfig:
    """OFDM / propagation parameters (Sec. II and V of the paper)."""

    Nc: int = 24          # number of subcarriers
    Nt: int = 14          # number of time symbols
    carrier_freq_hz: float = 2.1e9   # LTE Band 1(2110-2170MHz)처럼 흔한 FDD 하향링크 대역 임의로 지정
    subcarrier_spacing_hz: float = 15e3 # 흔히 쓰이는 15KHz 설정
    symbol_duration_s: float = 71.4e-6   # 1/15KHz에 CP 고려해서 (7/6)곱하기 
    num_paths: int = 200                 # 3GPP TR 38.901 Table 7.5-6 UMa NLOS는 정확히 N=20 클러스터 × M=20 ray/클러스터 = 400개이나 계산속도를 위해 200으로 해도 rank-1 붕괴 없음.
    # NOTE: 300ns is the textbook 3GPP 38.901 UMa-NLOS median RMS delay
    # spread at ~2GHz, but with only Nc=24 subcarriers at 15kHz spacing
    # (360kHz total observed bandwidth) that value makes the channel
    # essentially flat over the whole pilot band (coherence bandwidth
    # >> 360kHz), collapsing the frequency covariance to rank ~4 (99%
    # energy) and making every estimator (incl. plain averaging)
    # trivially near-perfect -- which would defeat the point of this
    # experiment. We scale the delay spread up so the 24-subcarrier
    # window exhibits genuine frequency selectivity, which is what
    # actually exercises the structured-covariance GMM estimators.
    #
    # However, this can't be scaled up arbitrarily: the pilot grid
    # itself (Npc=10, Npt=5, Np=50 out of D=Nc*Nt=336) is fixed by the
    # paper (Sec. V, Fig. 1) and is NOT something we're free to tune to
    # fit the channel -- unlike the channel model, which is already our
    # own substitute for QuaDRiGa (see README). If the delay spread is
    # pushed too high, the channel's covariance needs more effective
    # rank than this fixed 50-pilot lattice can resolve, and *every*
    # estimator -- including a genie one with perfect knowledge of the
    # instantaneous channel's PDP/DS -- hits a noise-independent MSE
    # floor no amount of SNR can beat (verified empirically: at the
    # previous value of 8000ns, effective rank(99%) = 37 and the
    # noiseless (SNR -> inf) genie floor for this exact pilot grid is
    # ~0.114 -- matching, almost exactly, the plateau seen in a full
    # Fig. 3 reproduction run whose curves stopped decreasing past
    # SNR=10dB instead of continuing down to ~1e-4 like the paper's).
    # A sweep of candidates against that same genie-floor computation
    # gives:
    #   300ns  -> rank99=4,  floor=3.0e-08
    #   800ns  -> rank99=7,  floor=1.6e-06
    #   1000ns -> rank99=8,  floor=1.1e-05
    #   1500ns -> rank99=10, floor=1.9e-04
    #   8000ns -> rank99=37, floor=1.1e-01  (too high, see above)
    # 1000ns keeps the floor ~9x below the paper's lowest plotted MSE
    # (1e-4 in Fig. 3's v=3km/h panel), leaving enough headroom that
    # the finite-sample/finite-K GMM fits (which sit above the
    # theoretical floor) still don't visibly plateau before 30dB, while
    # still being clearly frequency-selective (rank99=8 vs. 4 at the
    # textbook 300ns value) -- a much smaller, better-justified
    # inflation (3.3x textbook) than the previous 8000ns (26.7x
    # textbook), which is what pushed the floor above the visible range.
    rms_delay_spread_mean_s: float = 1000e-9  # see derivation above; was 8000e-9
    rms_delay_spread_std_db: float = 3.0      # 임의의 값


# Drop은 하나의 독립적인 시뮬레이션 환경 또는 단말의 특정 배치 위치 및 채널 상태 한 순간을 의미
def _draw_drop(rng: np.random.Generator, n_drops: int, cfg: SystemConfig, #난수 생성기(rng), 환경 개수(n_drops), 고정된 시스템 parameter들 (SystemConfig, 단말의 이동 속도(velocity_mps)를 입력받아 (복소 경로 이득, 지연 시간, 도플러 주파수, 전력 프로파일) 4개 배열을 반환하는 함수를 정의
                velocity_mps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:# 이 함수가 매번 호출때마다 랜덤생성하게 함수 밖에서 받아옴으로써 iteration 돌더라도 완전 다른 결과의 샘플들을 뱉어내려고.
    """Draw random multipath parameters for `n_drops` independent drops.

    Returns (gains, delays, doppler_freqs) each of shape (n_drops, L).
    """
    L = cfg.num_paths

    # log-normal RMS delay spread per drop (per-environment diversity)
    ds_db_std = cfg.rms_delay_spread_std_db
    ds = cfg.rms_delay_spread_mean_s * 10.0 ** (rng.normal(0.0, ds_db_std, size=(n_drops, 1)) / 20.0) # dB scale에서 normal dist.인 shadowing을 적용 

 
    delays = rng.exponential(scale=ds, size=(n_drops, L))  # n_drops by L 크기의 행렬로 delay 요소들이 exponential한 분포로 생성. 이때 delay spread의 RMS값은 Exp.dist.의 기대값이고 그걸 scale이라는 변수로 받아서 생성하는 것임. 
    delays.sort(axis=1)  # L=200 방향으로 각 Drop 내의 지연 시간들을 오름차순(시간 순)으로 정렬
    powers = np.full((n_drops, L), 1.0 / L)

    # random arrival angles -> per-path Doppler shift (classic Jakes model)
    angles = rng.uniform(0.0, 2 * np.pi, size=(n_drops, L))  # n_drops by L 의 unif dist.인 앵글 생성.
    doppler = (velocity_mps[:, None] / SPEED_OF_LIGHT) * cfg.carrier_freq_hz * np.cos(angles)

    # complex path gains g_l ~ CN(0, p_l)
    gains = (rng.normal(size=(n_drops, L)) + 1j * rng.normal(size=(n_drops, L))) # 위상반영
    gains *= np.sqrt(powers / 2.0) # 파워반영

    return gains, delays, doppler, powers


def generate_channel_dataset(
    n_samples: int,                                         # 전체 샘플수
    cfg: SystemConfig = SystemConfig(), 
    velocity_kmh: float | None = 3.0,
    velocity_range_kmh: tuple[float, float] | None = None,
    batch_size: int = 2000,                                 # 한번에 계산할 샘플수
    seed: int | None = None,                                # seed를 밖에서 받는 이유는 1. 학습/검증/테스트 데이터가 서로 겹치지 않게(독립적이게) 만들기 위해서, 2. 나중에 똑같은 seed로 다시 부르면 완전히 똑같은 데이터를 재현함으로써 troubleshooting 잘되게 하고자.
) -> np.ndarray:
    """Generate `n_samples` independent channel drops H[n, c, t].

    Exactly one of `velocity_kmh` (fixed velocity) or `velocity_range_kmh`
    (uniform random velocity range, e.g. (0, 300)) must be given.
    """
    if (velocity_kmh is None) == (velocity_range_kmh is None):
        raise ValueError("Specify exactly one of velocity_kmh or velocity_range_kmh")

    rng = np.random.default_rng(seed)
    Nc, Nt = cfg.Nc, cfg.Nt
    c_idx = np.arange(Nc)
    t_idx = np.arange(Nt)
    subcarrier_freqs = cfg.carrier_freq_hz + c_idx * cfg.subcarrier_spacing_hz

    H = np.empty((n_samples, Nc, Nt), dtype=np.complex128)

    n_done = 0
    while n_done < n_samples:
        n_b = min(batch_size, n_samples - n_done) # 한번에 처리할 샘플수. 앵간하면 배치수인 2000이나 마지막엔 좀 잘림

        if velocity_kmh is not None:
            v_mps = np.full(n_b, velocity_kmh / 3.6) # 2000개 길이 벡터에 m/s단위로 환산한 속도 다 채워
        else:
            lo, hi = velocity_range_kmh
            v_mps = rng.uniform(lo, hi, size=n_b) / 3.6

        gains, delays, doppler, _ = _draw_drop(rng, n_b, cfg, v_mps)

        # freq_phase[n, l, c] = exp(-1j*2*pi*f_c(c)*tau[n,l])
        freq_phase = np.exp(-1j * 2 * np.pi * subcarrier_freqs[None, None, :] * delays[:, :, None])
        # doppler_phase[n, l, t] = exp(1j*2*pi*f_D[n,l]*t*Ts)
        doppler_phase = np.exp(1j * 2 * np.pi * doppler[:, :, None] * t_idx[None, None, :] * cfg.symbol_duration_s)

        H_b = np.einsum('nl,nlc,nlt->nct', gains, freq_phase, doppler_phase) # H[n,c,t] = sum__ gains[n,l] × freq_phase[n,l,c] × doppler_phase[n,l,t]을 n,l,c에 대해 전부 case by 곱해서 3차원 채널 형성.
        H[n_done:n_done + n_b] = H_b # 최종 2D time, freq channel matrix 생성.
        n_done += n_b

    return H


def calibrate_normalization(cfg: SystemConfig = SystemConfig(), n_calib: int = 20000, seed: int = 0) -> float: #논문에서 ch power가 Nc*Nt되어야한다고 해서 그거 맞춰주는 용도
    """Compute a scalar so that E[||h||_2^2] = Nc*Nt exactly (paper's normalization)."""
    H = generate_channel_dataset(n_calib, cfg, velocity_kmh=None, velocity_range_kmh=(0.0, 300.0), seed=seed)
    mean_sq_norm = np.mean(np.sum(np.abs(H.reshape(n_calib, -1)) ** 2, axis=1))
    target = cfg.Nc * cfg.Nt
    return float(np.sqrt(target / mean_sq_norm))
