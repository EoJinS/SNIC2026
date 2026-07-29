# ASU EM 클러스터링 — 데이터셋 + 실행 코드

ASU Campus 3.5 GHz 채널(FFT 1024 중 **32개마다** 뽑은 32 서브캐리어)로
GMTC full-covariance GMM **EM 클러스터링**을 돌려 공간 구역 맵을 그리는 자립형 패키지.

```
asu_em/
├── README.md
├── run_em.py          ← 실행 스크립트 (이것만 돌리면 됨)
├── gmtc_em.py         ← GMTC EM 함수 (자립형, 외부경로 불필요)
├── data/
│   ├── asu_fd_32x32_stride32.npy   ← 채널 데이터
│   └── asu_pos.npz                 ← 유저/BS 위치 (맵용)
└── output/            ← 결과 PNG 저장 위치
```

## 1. 데이터셋

**`data/asu_fd_32x32_stride32.npy`** — shape `(21332, 1, 32, 32)`, `complex64`
- 축: `(user, 1, antenna=32, subcarrier=32)` → `[:, 0]` 하면 `(21332, 32, 32)`
- **서브캐리어 선택**: FFT 1024 중 `[0, 32, 64, …, 992]` = **32개마다** → 15.36 MHz **전대역**(주파수 다양성 보존)
- BS 32-안테나 ULA(0.5λ), UE 1, SCS 15 kHz, num_paths 25
- 유저 21,332명 (active ∩ uniform 2 m 격자), **PCN 정규화**(채널별 ‖h‖=1)

**`data/asu_pos.npz`** (맵 그릴 때 필요)
- `pos` `(21332,3)` 유저 좌표 `[x y z]` — **채널과 행 순서 동일**
- `tx` `(3,)` BS 좌표
- `pos_nouser` `(46774,3)` 유저 없는 격자점(건물/차폐) — 맵 배경 회색점

## 2. EM 파이프라인 (`run_em.py`)

1. `H (T,32,32)` → real∥imag concat → `(T, 2048)`
2. **N=256 청크**로 분할 → 유저당 8청크 → `(170656, 256)`
3. **GMTC full-cov GMM EM** (`gmm_em_fit_fullcov`, 50 iter, seed=0)
4. 청크 라벨 → **유저 라벨 = 그 유저 8청크의 다수결**
5. 유저 위치를 라벨색으로 맵 산점도 (회색=No user, 빨간별=BS)

## 3. 실행

```bash
# 의존: torch, numpy, matplotlib  (GPU 있으면 자동 사용, 없으면 CPU)
python run_em.py                    # 기본 K=4,8,16 → output/em_map_*_K{4,8,16}.png
python run_em.py --Ks 8 16 32       # K 지정
CUDA_VISIBLE_DEVICES=0 python run_em.py   # 특정 GPU
```

옵션: `--data`(다른 채널 npy), `--pos`, `--N`(청크크기, 기본 256), `--Ks`, `--em_iter`, `--outdir`.

## 4. 참고

- `gmtc_em.py`는 GMTC `main.py`에서 EM에 필요한 함수만 추출한 것이라 **GMTC 저장소가 없어도 동작**합니다.
- GPU 없으면 자동으로 CPU에서 돌아갑니다(큰 K는 느릴 수 있음).
- 결과 맵은 `output/`에 저장됩니다.
