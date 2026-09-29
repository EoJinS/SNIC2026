target directory: /home/ejseo/SNIC2026/Share/pipeline
context : /home/ejseo/SNIC2026/Share/README.md

TX ULA 1x4 에서 space-frequency 실험 뿐만 아니라 velocity를 바꿔가면 time-frequency 축도 보아야 해. ULA 1x4에서 time-freq 와 space-freq 를 보려면 3차원 정보가 필요한데, 우선은 time-freq 2차원 그래프, space-freq 2차원 그래프를 각각 생성해줘. 아래 조건을 반영해서 실험을 진행해줘. 필요 시 진행상황 보고하기

1. GMM structure 중 b-toep 는 제외하고 실험하기. plot 시 그래프 및 legend 등 b-toep 는 아예 삭제하고 plot 하기.(실험 시간이 너무 오래걸리기 때문에)
2. velocity = 3m/s, 6m/s, 10m/s 로 실험해
3. parameter 는 @SNIC2026/Share/pipeline/generate_channels_subregion100_marray.py 와 같이 하기. (fc, SCS, TX ULA 등)
4. 실험 시간이 오래 걸릴 것으로 예상되므로 GPU 가속 사용하기
5. 실험은 SCALES= paper50 으로 돌리기
6. plot 시 SCALES(paper50) 은 제목에서 삭제하기.
7. 실험 내용 및 변경 사항은 /home/ejseo/SNIC2026/Share/README.md에 정리하기. (실험 실행 방법, 생성 png 위치 등 명시하기)

---

## 실행 노트 (2026-09-16, 자동 실행 시 해석/보완)

이 guideline은 "ULA 1x4에서 3차원(안테나 x 주파수 x 시간) 정보가 필요하지만
우선 2차원 그래프 두 개(time-freq, space-freq)를 따로 만든다"는 취지였음.
**처음에는 space-freq(1x4 ULA, 기존 정적 M4 데이터)와 time-freq(SISO 단일
안테나)를 서로 다른 두 시스템에서 뽑아 실행했다가, 사용자가 "두 그래프 모두
동일한 1x4 ULA 시스템에서 나와야 한다"고 정정함.** SISO 단일 안테나는 물리
파라미터(fc/SCS/TX 위치)는 같아도 배열 중심에서 반파장 단위로 위치가 달라
멀티패스 위상이 달라지므로 "같은 1x4 ULA의 한 소자"와 동일하지 않음.

그래서 새 스크립트 `generate_channels_subregion100_ula_tf.py`로 **안테나(4) x
주파수(64) x OFDM심볼(14) 3차원 큐브**를 위치별로 한 번에 ray-trace함 (TX 배열은
marray 스크립트와 동일한 `PlanarArray(1,4)`, 시간/도플러 메커니즘은 SISO
time-freq 스크립트와 동일한 `Receiver(velocity=...)` + analytic-Doppler
`cfr(num_time_steps=14)`). `gmmce/pipeline.py`의 GMM-CE 추정기가 2차원(Nc,Nt)
축만 지원하므로, `derive_ula_tf_slices.py`가 이 큐브에서 기존 두 실험
드라이버가 바로 쓸 수 있는 2차원 슬라이스 두 종류를 뽑아냄:

- **space-frequency (안테나 x 주파수)**: OFDM 심볼 0 슬라이스. velocity와
  무관함(t=0에는 아직 도플러 위상이 쌓이지 않음 -- v=0 큐브의 symbol 0과 v=3
  큐브의 symbol 0을 수치로 직접 비교해 확인함, float32 노이즈 수준으로 일치)
  -> `--speed 0` 큐브 하나에서만 뽑아 `Data/{train,test}_M4_F64_subregion100_ulatf.npz`
  로 저장, 한 번만 실험 실행.
- **time-frequency (주파수 x OFDM 심볼)**: 고정된 TX 안테나 인덱스(0번)의
  슬라이스, velocity(0/3/6/10 m/s) 각 큐브에서 하나씩 ->
  `Data/{train,test}_siso_S14_F64_v{v}_subregion100_ulatf.npz` (기존
  `gmmce/subregion_tf_data.py` 로더가 기대하는 레이아웃과 동일하게 저장, 코드
  변경 없이 재사용 가능).
- 기존 실행 스크립트 2개에 `--data-tag` 옵션을 추가해 이 `_ulatf` 접미사가
  붙은 파일을 로드하도록 함 (`gmmce/subregion_data.py`,
  `gmmce/subregion_tf_data.py`). 기존 `train_M4_F64_subregion100.npz`(정적,
  시간축 없음)와 `train_siso_..._v10...npz`(진짜 단독 SISO)는 그대로 두고
  건드리지 않음 -- 새 `_ulatf` 데이터가 이 guideline이 요구하는 "하나의 1x4
  ULA 시스템"용 데이터.
- **b-toep 제외(조건 1)**: 두 실행 스크립트에 `--no-btoep` 플래그를 추가함
  (`gmmce/pipeline.py`의 `train_all`은 건드리지 않고, 스크립트 쪽에서 학습
  대상 variant 목록과 legend/plot 순서에서만 제외 -- 다른 실험에서 b-toep이
  필요하면 플래그 없이 그대로 쓸 수 있음).
- **제목에서 SCALES(paper50) 삭제(조건 6)**: `--scale paper50`일 때만 플롯
  제목의 `[paper50: K=..., N=...]` 접미사를 생략하도록 수정 (quick 등 다른
  스케일은 기존 동작 유지).
- 결과 디렉토리: `results_subregion_M4_ulatf_paper50/` (space-freq),
  `results_subregion_tf_v{0,3,6,10}_ulatf_paper50/` (time-freq, velocity별,
  v=0은 도플러 없는 기준선). 자세한 실행 커맨드와 PNG 위치는 README.md 참고.

### 완료 (2026-09-17)

5개 실험(space-freq 1개 + time-freq x velocity 4개) 모두 paper50 스케일,
GPU, `--no-btoep`로 정상 종료. 각 3개 플롯(nmse_vs_snr/training/components)
x json 생성 확인. 결과/재현 커맨드/수치 요약은
`/home/ejseo/SNIC2026/Share/README.md`의
"## 1x4 ULA velocity sweep: space-freq + time-freq as two slices of ONE
system" 섹션 참고.

추가로 동일 5개 실험을 `--comb-spacing 8`로도 재실행 완료 (같은 데이터,
`results_subregion_M4_ulatf_comb8_paper50/`,
`results_subregion_tf_v{0,3,6,10}_ulatf_comb8_paper50/`). 요약은 README의
"### Comb-spacing-8 sweep" 참고 -- space-freq는 `full`이 전 SNR에서 최고,
time-freq는 30dB에서 `2x1D`/`2x1D-toep`가 `full`을 근소하게 앞서는(b-toep
제외 상태에서 그 역할을 `2x1D` 계열이 대신 맡는) 결과.