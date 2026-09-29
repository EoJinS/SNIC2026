# 1x4 ULA velocity sweep: 실험 세팅 정리

"space-freq + time-freq as two slices of ONE system" (README.md 330행~) 실험의 pilot 패턴, 데이터 생성, 실험 세팅을 코드 기준으로 정리한 문서.
근거 코드: `pipeline/generate_channels_subregion100_ula_tf.py`, `pipeline/derive_ula_tf_slices.py`,
`pipeline/GMM_CE/run_subregion_experiments.py`, `pipeline/GMM_CE/run_subregion_tf_experiments.py`,
`pipeline/GMM_CE/gmmce/{pilots,pipeline,subregion_data,subregion_tf_data}.py`.
(코드를 읽고 정리한 것이며, 실험을 새로 돌려 검증한 것은 아님.)

---

## 0. 전체 흐름 (한눈에)

```
POSTECH scene 레이트레이싱 (위치×속도별)          generate_channels_subregion100_ula_tf.py
  -> cube h[p] : (M=4 antenna, S=14 symbol, F=64 subcarrier)
        |
        |  derive_ula_tf_slices.py  (2-D 슬라이스)
        +-- space-freq (antenna x freq)  : v=0 cube의 symbol-0 슬라이스   -> D=256
        +-- time-freq  (freq x symbol)   : 속도별 cube의 antenna-0 슬라이스 -> D=896
        |
        v
SubregionChannels / SubregionChannelsTF : reshape, 전역 정규화, train/calib/test 분할
        |
        v
train_all(Htr) : 깨끗한 채널 vec(H)로 GMM EM 학습 (pilot/노이즈/속도 정보 없음)
        |
        v
평가 : y = A·vec(H) + n  (A=pilot 선택행렬, n~CN(0, σ²I)) -> GMM conditional mean -> NMSE
```

---

## 1. Pilot pattern (comb-8)

**정의:** `make_grid` (`run_subregion_tf_experiments.py:98`, `run_subregion_experiments.py:98`).
`--comb-spacing 8` 이면 다음과 같다.

- **freq 축:** `pilot_carriers = arange(0, Nc, 8)`. Nc=64 이므로 subcarrier 0, 8, 16, …, 56, 즉 Npc=8.
- **time 축 (time-freq):** `pilot_symbols = arange(Nt)`. 14개 OFDM symbol 전부에 pilot. freq는 sparse, time은 dense.
- **antenna 축 (space-freq):** 같은 코드에서 "symbol" 자리에 antenna가 들어가므로 4개 안테나 모두 관측.
- **separable 구조:** Cartesian product 격자 (A = A_t ⊗ A_c) 이므로 Kron/2x1D 추정기의 가정이 유지된다.
- **관측 모델:** A는 0/1 selection matrix. 행은 symbol-major 순서, 열 index는 `t*Nc + c` (`pilots.py:selection_matrix`).
  벡터화는 column-major (`vec(H)[t*Nc+c] = H[c,t]`, freq가 fastest).

| | 전체 D | full pilot (`A=I`) | comb-8 Np |
|---|---|---|---|
| time-freq (freq × symbol) | 64×14 = 896 | 896 | 8×14 = **112** |
| space-freq (freq × antenna) | 64×4 = 256 | 256 | 8×4 = **32** |

```
time-freq, 행=OFDM symbol(14), 열=subcarrier(64), ■=pilot
sym  1 :  ■.......■.......■.......■.......■.......■.......■.......■.......
sym  2 :  ■.......■.......■.......■.......■.......■.......■.......■.......
  ...      (14개 symbol 모두 동일한 위치: subcarrier 0,8,16,...,56)
sym 14 :  ■.......■.......■.......■.......■.......■.......■.......■.......
```

- comb-8에서는 시간축 보간 없이 freq 방향만 보간하면 된다.
- 기본(full pilot) 실험은 `A=I` (모든 원소 관측).
- 참고: `results_..._circ_c8t2`, `results_..._unif_c8t2` 폴더는 `run_subregion_tf_pilot_experiments.py` /
  `plot_pilot_patterns.py` (uniform comb vs circular comb) 계열의 별도 pilot 실험이다. 이 sweep의 comb-8은 위의 단순 comb.

---

## 2. Training / test 데이터 생성

### (a) Ray-tracing: `generate_channels_subregion100_ula_tf.py`

`--speed {0,3,6,10}` 마다 한 번씩 실행. 위치별로 `(M=4 안테나, S=14 symbol, F=64 subcarrier)` cube 생성.

| 항목 | 값 |
|---|---|
| 장면 | POSTECH scene, sionna.rt `PathSolver` |
| 반송파 fc | 3.5 GHz (λ ≈ 8.57 cm) |
| TX 위치 | (-70, -25, 15) |
| TX 어레이 | `PlanarArray(1, 4)`, 간격 0.5λ, iso, V-pol |
| RX | 안테나 1개, iso, V-pol, 높이 1.5 m |
| 레이트레이싱 옵션 | LOS, specular, diffuse, refraction 모두 on, max_depth=5, samples_per_src=1e6, seed=42 |
| 산란계수 | itu_concrete 0.1, itu_brick 0.15, itu_very_dry_ground 0.05 |
| 주파수축 | F=64, SCS=240 kHz (`subcarrier_frequencies(64, 240e3)`), 총 대역폭 15.36 MHz |
| 시간축 | S=14 symbol, symbol_dt=71.4 µs, slot = 0.9996 ms (≈1 ms) |
| CFR 옵션 | `normalize_delays=False`, `normalize=False` |

- **시간축 생성 방식:** 위치당 `PathSolver` 호출은 1번. Sionna의 analytic Doppler를 사용한다
  (`Receiver(velocity=...)` + `paths.cfr(sampling_frequency=1/71.4µs, num_time_steps=14)`).
  symbol마다 레이트레이싱을 다시 하지 않는다.
- **속도:** 위치마다 `v = speed·(cosθ, sinθ, 0)`. heading θ는 위치마다 독립적으로 U[0, 2π)에서 추출.
  seed는 `seed+1000` (train) / `seed+1001` (test). `--heading`으로 고정도 가능하나 이 sweep은 랜덤 heading.
  speed=0 이면 v=0, Doppler 없음 (14개 symbol 동일).
- **최대 Doppler / slot 위상:**

  | v (m/s) | f_D = v/λ (Hz) | slot 위상 2π·f_D·slot |
  |---|---|---|
  | 0 | 0 | 0° |
  | 3 | ≈ 35 | ≈ 12.6° |
  | 6 | ≈ 70 | ≈ 25.2° |
  | 10 | ≈ 117 | ≈ 42° |

- **위치 grid:** `Data/positions_subregion_grid_0.25m_100x100.npz`.
  x∈[-80,20], y∈[-60,40], 0.25 m 간격 격자에서 건물 margin을 제외한 **123,583개** 후보 위치 (높이 1.5 m).
- **Train/test 분할:** seed=42 permutation 후 앞 **110,000개 = train**, 다음 **12,000개 = test**.
  두 집합은 위치가 겹치지 않고, 각각 sort해서 사용한다. permutation은 속도와 무관하게 고정이므로
  4개 속도가 정확히 같은 위치 집합을 쓴다. 즉 속도만 변수.
- **저장:** `h` : `(N, M·S·F)` complex64, index = `m*S*F + t*F + f` (antenna-major, symbol, subcarrier-minor).
  파일명 `Data/{train,test}_ula4_S14_F64_v{v}_subregion100.npz` (train ≈ 3.1 GB, test ≈ 345 MB).
  속도 4개 합쳐 약 14 GB. 레이트레이싱 속도 ≈ 150 pos/s (속도당 train ≈ 12–13분, test ≈ 1.5분).
- **chunk:** drjit 2^32 텐서 한계 때문에 `chunk·M·S·F·paths < 2^32` 가 되도록 chunk를 자동 축소한다 (SISO 대비 약 4× 작음).

### (b) 2-D 슬라이싱: `derive_ula_tf_slices.py`

GMM-CE 라이브러리는 2-D `(Nc, Nt)` 쌍만 받으므로, 하나의 cube에서 2개의 2-D 슬라이스를 만든다.

- **space-freq (antenna × freq):** v=0 cube의 symbol 0 슬라이스 → `(N, 4·64=256)`,
  `Data/{train,test}_M4_F64_subregion100_ulatf.npz`. t=0 에서는 Doppler 위상이 0 이라 속도와 무관하다
  (v=0 과 v=3 의 symbol-0 슬라이스가 ~5e-7 차이로 일치함을 README가 확인했다고 기록). 한 번만 생성.
- **time-freq (freq × symbol):** 속도별 cube의 TX 안테나 0번 슬라이스 → `(N, 14·64=896)`,
  `Data/{train,test}_siso_S14_F64_v{v}_subregion100_ulatf.npz`. 4개 속도 각각 한 파일.
- 독립된 SISO 안테나가 아니라 **같은 4-소자 어레이의 한 소자**라는 점이 핵심 (sub-wavelength 소자 간격 때문에 multipath 위상이 다름).

### (c) 로딩과 정규화

`SubregionChannels` (space-freq, `subregion_data.py`), `SubregionChannelsTF` (time-freq, `subregion_tf_data.py`).

- **reshape:** `(N, S, F)` → `(N, Nc=64, Nt=14)` 로 transpose, complex128. (space-freq는 `(N, M, F)` → `(N, Nc=64, Nt=M=4)`.)
- **정규화:** train pool 전체에서 실수 스칼라 하나를 구해 `E‖vec(H)‖² = Nc·Nt` 가 되도록 스케일링.
  같은 스칼라를 test에도 적용. 샘플별이 아닌 **전역 스케일**이며, 속도별 train pool에서 각각 계산된다.
- **분할 (seed=0 permutation):** train은 permute. test pool(12,000개)은 permute 후 절반씩 나눠
  앞 6,000개가 test 후보, 뒤 6,000개가 calibration 후보.
- **paper50 사용량:**

  | 용도 | 개수 | 출처 |
  |---|---|---|
  | train | 100,000 | train pool 110,000 중 |
  | calib | 4,000 | test pool 뒤 절반 |
  | test | 5,000 | test pool 앞 절반 |

### (d) 학습·평가 입력

- **학습 입력:** `Xtr = vec(H_train)` : `(100000, D)`, D = 896 (time-freq) / 256 (space-freq).
  깨끗한 채널 값만 사용하며 pilot, 노이즈, 속도 정보는 학습에 들어가지 않는다.
  속도별로 GMM을 **독립적으로 4번** 학습하며, 속도를 조건으로 받는 모델은 없다.
- **평가 입력:** 테스트 채널 `H_te`에 대해 `y = A·vec(H) + n`, `n ~ CN(0, σ²I)`.
  `σ² = 10^(-SNR/10)`. 채널이 평균 전력 1/원소로 정규화되어 있으므로 원소 기준 SNR.

---

## 3. 실험 세팅

### 3.1 실험 매트릭스

| 구분 | 데이터 | D | 실행 스크립트 | 결과 폴더 |
|---|---|---|---|---|
| space-freq (1x4 ULA, antenna×freq) | `*_M4_F64_..._ulatf.npz` | 256 | `run_subregion_experiments.py --m 4` | `results_subregion_M4_ulatf[_comb8]_paper50/` |
| time-freq (freq×symbol), v=0,3,6,10 | `*_siso_S14_F64_v{v}_..._ulatf.npz` | 896 | `run_subregion_tf_experiments.py --speed v` | `results_subregion_tf_v{v}_ulatf[_comb8]_paper50/` |

pilot 설정 2가지 (full pilot, comb-8) × [space-freq 1 + time-freq 4 속도] = 총 10회 실행.

### 3.2 공통 하이퍼파라미터 (`--scale paper50`, Fesl et al. Sec. V 설정)

| 항목 | 값 |
|---|---|
| K (joint GMM 성분 수) | 128 |
| Kt, Kc (kron 용 1-D GMM) | 8, 16 (Kt·Kc = 128) |
| Kt_2x1d, Kc_2x1d (2x1D cascade 용) | 32, 96 (합 = 128) |
| EM 반복 n_iter | 50 |
| n_train / n_calib / n_test | 100,000 / 4,000 / 5,000 |
| SNR 리스트 | {-10, 0, 10, 20, 30} dB |
| seed | 0 (EM 초기화, permutation) |
| 연산 | GPU (torch), 기본 ON (`--gpu`) |
| 정밀도 | 채널 complex128, 저장은 complex64 |
| 옵션 | `--no-btoep`, `--data-tag ulatf`, (comb 실험) `--comb-spacing 8` |

### 3.3 추정기 (비교 대상)

`--no-btoep` 이므로 `GMM b-toep` 은 학습도 플롯도 하지 않는다. (Barton–Fuhrmann Toeplitz M-step이 다른 변형보다
약 10배 많은 EM 반복을 필요로 해서 런타임 비중이 크고 이 sweep에서는 얻는 것이 적다는 판단.)

| 추정기 | 설명 |
|---|---|
| GMM full | D 차원 unconstrained 복소 GMM (K=128), EM 50회 |
| GMM kron | freq(Nc) GMM × time(Nt) GMM 의 Kronecker 결합 (Kt·Kc = 128 성분) |
| GMM b-circ | 각 성분 공분산을 block-circulant 로 제약 |
| GMM 2x1D | freq 1-D GMM + time 1-D GMM 의 cascade (calib set 으로 보정) |
| GMM 2x1D-toep | 2x1D 의 각 1-D GMM 을 Toeplitz 제약 |
| GMM 2x1D-circ | 2x1D 의 각 1-D GMM 을 circulant 제약 |
| PDP+DS kron / 2x1D | genie baseline (PDP + Doppler spectrum 기반 공분산, 학습 없음) |

- 추정: pilot 관측 `y = A h + n` 에서 GMM의 conditional-mean estimator (CME).
- 지표: NMSE = `‖Ĥ − H‖² / ‖H‖²` 의 평균 (`normalized_mse`).

### 3.4 세 가지 평가 (각 결과 폴더에 PNG + JSON)

1. **`nmse_vs_snr`:** 전체 모델(K=128, N=1e5)을 한 번 학습하고 SNR {-10,0,10,20,30} dB 에서 평가.
   SNR별 노이즈 seed = `seed + 5000 + snr`. 9개(b-toep 제외 시 8개) 추정기 모두.
2. **`nmse_vs_training`:** SNR = 10 dB 고정. n_train ∈ {5,000, 12,500, 33,333, 100,000}
   (= n_train/20, /8, /3, /1). 각 n_train마다 재학습. 추정기는 GMM 6종 (PDP+DS 제외).
3. **`nmse_vs_components`:** SNR = 10 dB 고정. K ∈ {8, 32, 64, 128}
   (= max(2,K/16)→8, max(4,K/4)→32, K/2, K). 각 K에서 `Kt = round(√K)`, `Kc = K // Kt`,
   `Kt_2x1d = round(K/4)`, `Kc_2x1d = K − Kt_2x1d` 로 재학습. 추정기는 GMM 6종.

### 3.5 재현 커맨드

```bash
# 1) 채널 생성 (pipeline/, GPU 2 사용)
for v in 0 3 6 10; do
  CUDA_VISIBLE_DEVICES=2 python3 -u generate_channels_subregion100_ula_tf.py --speed $v
done
# 2) 2-D 슬라이스
python3 derive_ula_tf_slices.py --m 4 --speeds 0 3 6 10 --antenna 0

# 3) 실험 (pipeline/GMM_CE/)
# full pilot
python3 run_subregion_experiments.py --m 4 --scale paper50 --no-btoep --data-tag ulatf --gpu
for v in 0 3 6 10; do
  python3 run_subregion_tf_experiments.py --speed $v --scale paper50 --no-btoep --data-tag ulatf --gpu
done
# comb-8 pilot
python3 run_subregion_experiments.py --m 4 --scale paper50 --no-btoep --data-tag ulatf --comb-spacing 8 --gpu
for v in 0 3 6 10; do
  python3 run_subregion_tf_experiments.py --speed $v --scale paper50 --no-btoep --data-tag ulatf --comb-spacing 8 --gpu
done
```

소요 시간 (GPU, `--no-btoep`, README 기록): space-freq full pilot ≈ 71분 / comb-8 ≈ 66분.
time-freq full pilot ≈ 3.4–4.6 h / 속도 (v=0 최단, v=10 최장), comb-8 ≈ 3.0–4.3 h / 속도.
comb pilot은 학습 비용을 바꾸지 않는다 (EM은 전체 채널로 학습하고, CME 평가 단계에서만 sparse pilot을 본다).

---

## 4. 결과 요약 (README 기록, K=128, N_train=1e5, n_iter=50, 30 dB)

**full pilot**
- space-freq (D=256): full 1.4e-4 < 2x1D 2.1e-4 < 2x1D-toep 2.4e-4 < kron 3.1e-4 < 2x1D-circ 4.8e-4 < b-circ 6.0e-4 < PDP+DS 2x1D 7.7e-4 < PDP+DS kron 1.0e-3.
- time-freq (D=896), full / kron NMSE:

  | v (m/s) | full | kron |
  |---|---|---|
  | 0 | 2.1e-5 | 2.4e-5 |
  | 3 | 2.8e-5 | 3.3e-5 |
  | 6 | 3.2e-5 | 4.0e-5 |
  | 10 | 3.6e-5 | 4.4e-5 |

  속도가 커질수록 모든 추정기에서 NMSE가 매끄럽게 단조 증가한다.
  (full pilot 이라 시간 보간이 필요 없으므로 aliasing/보간 효과가 아니라 Doppler spread 증가로 GMM prior가 구조를 잡기 어려워지는 효과.)

**comb-8 pilot**
- b-circ, 2x1D-circ, PDP+DS 는 DFT/circulant 지연 모델의 aliasing 때문에 hard NMSE floor (SNR ≳ 10–20 dB 부터 평탄화).
- space-freq: full 이 모든 SNR 에서 best (30 dB: full 2.0e-3, 2x1D 4.3e-3, 2x1D-toep 4.8e-3, kron 8.6e-3, 2x1D-circ 2.0e-2, b-circ 3.2e-2, PDP+DS ≈ 7.7–8.5e-2).
- time-freq: -10..20 dB 에서는 full/kron 이 앞서지만 30 dB 에서는 모든 속도에서 2x1D 가 full 을 추월
  (v=10: 2x1D 3.5e-3 < 2x1D-toep 3.9e-3 < full 4.0e-3 < kron 8.1e-3; v=0: 2x1D 3.2e-3 < full 3.5e-3 < 2x1D-toep 4.1e-3).

---

## 5. 주의 사항

- **time-freq 벡터 순서:** 저장된 벡터는 symbol-major (`t*F+f`) 이고 GMM 벡터화도 `t*Nc+c` 라 index 는 일치하지만,
  저장 축은 (symbol, freq), 라이브러리가 기대하는 축은 (freq, symbol) 이라 transpose 가 필요하다.
- **space-freq 의 두 번째 축은 antenna:** 논문의 OFDM symbol 축을 antenna 로 대체한 proxy.
- **v=0 time-freq 와 space-freq 는 같은 raw 데이터의 서로 다른 index:** v=0 의 antenna-0 슬라이스(모든 symbol)와
  symbol-0 슬라이스(모든 antenna)는 독립적인 2-D 뷰다.
- **GMM 은 속도를 모른다:** 속도는 어떤 데이터 pool 을 로드할지만 결정한다. EM 수식에는 속도 입력이 없다.
- **b-toep 관련 README와 코드의 불일치:** README는 Toeplitz 반복 수 기본값을 `max(10*n_iter, 120)` 이라 적었으나,
  `gmmce/pipeline.py` 의 docstring/코드는 `max(3*n_iter, 120)` 이다. 이 sweep은 `--no-btoep` 이므로 결과에는 영향이 없다.
- **SNR 정의:** 채널이 평균 전력 1/원소로 정규화되어 있으므로 `σ² = 10^(-SNR/10)` 이 곧 원소 기준 SNR (pilot 전력 = 1, 노이즈 CN(0, σ²)).
