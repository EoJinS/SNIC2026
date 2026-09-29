Please run an additional experiment for the Weichselberger Model structure under the GMM configuration in the README.md experiment titled "1x4 ULA velocity sweep: space-freq + time-freq as two slices of ONE system".

Please follow these conditions:

Do NOT re-run other structures that have already been completed (e.g., Toeplitz, full GMM, b-toep, etc.).

Run the Weichselberger Model experiment using the exact same parameter settings as the existing GMM structures. (experiment setting context : /home/ejseo/SNIC2026/Share/expsetting.md )

Add and update the experiment result plots to incorporate the new Weichselberger Model results in the following directories:

/home/ejseo/SNIC2026/Share/pipeline/GMM_CE/results_subregion_tf_v{0,3,6,10}_ulatf_{circ, unif}_c8t2_paper50

/home/ejseo/SNIC2026/Share/pipeline/GMM_CE/results_subregion_M4_ulatf_comb8_paper50

---

## 실행 노트 (2026-09-29, 자동 실행 시 해석/보완)

**Structure spec.** 이 guideline 자체에는 Weichselberger 구조를 어떻게 구현할지
수식이 없었음. 최초 시도는 "b-circ처럼 데이터에서 뽑은 고정 basis 하나 + 대각
EM"이었는데, 사용자가 실행 중 `Share/Weichselberger_GMM_OptionB_Experiment_Guide.pdf`
("Option B: Component-wise Tx/Rx Eigenbasis Learning")를 지정하며 이 basis가
가이드와 맞는지 확인하라고 지시함. 대조 결과 guide는 **component마다 별도의
Tx/Rx eigenbasis**를 매 EM iteration마다 responsibility-weighted marginal
covariance에서 다시 뽑고, coupling matrix Omega_k는 그 basis에 대한 직접 moment
추정치로 갱신하는 알고리즘(Algorithm 1)을 명시하고 있어 최초 구현과 달랐음.
지시에 따라 실험을 멈추고, 이미 갱신했던 9개 결과 디렉토리를 백업에서 복구한 뒤
(다른 구조의 JSON 키는 전혀 건드리지 않았음을 diff로 확인), guide의 수식대로
`fit_weichselberger_gmm`을 CPU/GPU 양쪽에 다시 구현하고 (`Weichselberger_GMM_
OptionB_Experiment_Guide.pdf` sanity-check 항목: unitary power preservation,
marginal consistency 모두 수치로 검증) 처음부터 재실행함. 자세한 알고리즘/코드
변경/결과는 `Share/README.md`의 "`GMM Weichselberger` addition" 섹션 참고.

**"다른 구조 재실행 금지" 준수.** `run_weichselberger_addon.py`는
`train_all(..., which=("Weichselberger",))`만 호출해 다른 estimator를 전혀
다시 학습하지 않고, 기존 JSON에 `"GMM Weichselberger"` 키만 추가함 (9개
디렉토리 x 3개 JSON 파일 전부 diff로 확인: 다른 key는 단 하나도 바뀌지 않음).
파라미터도 각 결과 폴더의 기존 JSON에서 그대로 읽어와 재사용했음 (scale=paper50,
K=128, N_train=1e5, n_iter=50, seed=0, SNR 리스트 등 새로 정하지 않음).

**결과 요약.** `GMM Weichselberger`는 space-freq(M4 comb8)와 time-freq(4개
속도 x unif/circ 두 패턴) 전부에서 `kron`, `b-circ`보다 뚜렷이 낫고, `full`과
거의 같거나(circ 패턴) 오히려 더 나음(unif 패턴, 30dB 기준 4개 속도 모두). 수치
표는 README 참고.

## 실행 노트 2 (2026-09-29, full-pilot 확장)

사용자가 동일 실험을 full-pilot(comb/c8t2 아닌 원래 `results_subregion_M4_
ulatf_paper50/`, `results_subregion_tf_v{0,3,6,10}_ulatf_paper50/` 5개 디렉토리)
조건에도 추가해 달라고 요청. `run_weichselberger_addon.py`에 `m4fullpilot`/
`tffullpilot` 타겟을 추가해 동일한 "다른 구조 재실행 금지" 원칙으로 실행 (diff로
5개 디렉토리 x 3개 JSON 전부 확인: `"GMM Weichselberger"` 키만 추가됨).

실행 중 실제 버그 하나 발견 및 수정: full-pilot + 30dB에서만 `Weichselberger`가
`full`은 물론 `kron`에게도 짐 (다른 SNR에서는 정상적으로 이김) -- 이 저장소가
이미 `GMM b-toep`에 대해 문서화한 것과 동일한 complex64(fp32) 정밀도 한계
(full-pilot일 때 CME의 행렬이 D×D 풀사이즈라 30dB의 작은 노이즈에서 fp32가
불안정해짐). `GMMCE_GPU_STRUCT_DTYPE=complex128`로 nmse_vs_snr만 재적합해서
확인 및 수정함 (v=10: 6.09e-5 -> 3.47e-5, `full`의 3.55e-5와 거의 일치). comb8/
c8t2(sparse pilot)는 CME 행렬이 훨씬 작아(Np=32~56) 이 버그의 영향을 받지 않음
(재확인 완료, 그대로 둠). 자세한 내용은 README의 "Full-pilot extension" 참고.