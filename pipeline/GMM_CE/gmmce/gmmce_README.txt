[가장 아래: 재료]
  pilots.py          (파일럿 격자, vec/unvec)      ─┐
  structured.py      (DFT/Toeplitz 행렬)           ─┤  ← 다른 파일에 의존 안 함
  channel_model.py   (합성 채널 생성기)            ─┘

        │ 이 셋을 갖다 씀
        ▼
[1층: 알고리즘 부품]
  complex_gmm.py  ← structured.py 사용
      (fit_full_gmm / fit_toeplitz_gmm / fit_circulant_gmm)

        │
        ▼
[2층: "학습된 GMM으로 뭘 할지" 부품]
  cme.py          ← complex_gmm.py + pilots.py 사용   (식 2-3: 추정하기)
  kron_combine.py ← complex_gmm.py 사용               (GMM 두 개 합치기)
  pdp_ds.py       ← structured.py + pilots.py 사용    (베이스라인, GMM 안 씀)

        │
        ▼
[3층: 조합]
  cascade.py      ← complex_gmm.py + cme.py + pilots.py 사용   (2×1D 이어붙이기)

        │
        ▼
[4층: 전부 갖다 쓰는 배선]
  pipeline.py     ← 위의 거의 모든 파일을 import
                     (train_all: 6종 GMM 다 학습 / evaluate_joint, evaluate_cascade: MSE 계산)

        │
        ▼
[5층: 논문의 Fig.2/3/4 로직 -- 합성 채널판, ../../legacy/pipeline/GMM_CE/로 이동됨]
  experiments.py  ← pipeline.py + channel_model.py + pilots.py + (cme, kron_combine, pdp_ds, cascade 일부 직접 사용)
                     (run_fig2, run_fig3, run_fig4_training_size, run_fig4_components)

        │
        ▼
[6층: 진짜 실행 파일 = "메인실험코드" -- 합성 채널판, ../../legacy/pipeline/GMM_CE/로 이동됨]
  run_experiments.py  ← experiments.py + plotting.py
                     (CLI 인자 읽고 → run_fig2/3/4 호출 → 결과를 plotting.py로 그림)

[대신 이 프로젝트가 실제로 쓰는 6층: subregion100 (POSTECH 레이트레이싱) 채널판]
  run_subregion_experiments.py (../, GMM_CE/ 바로 아래)
      ← pipeline.py + subregion_data.py + pdp_ds.py + plotting.py
        (합성 channel_model.py 대신 subregion_data.py가 실측 채널을 공급)