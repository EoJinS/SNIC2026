# GMM-CE on subregion100

Structured-covariance GMM channel estimation (Fesl et al., Asilomar 2022:
`full` / `b-toep` (block-Toeplitz) / `b-circ` (block-circulant) / `kron` /
`2x1D` / `2x1D-toep` / `2x1D-circ`) evaluated on **position-only** channels
ray-traced with Sionna RT over a dense 0.25 m grid of the POSTECH campus
("`subregion100`": `x in [-80,20]`, `y in [-60,40]`).

The channel is a 1xM TX ULA x F=64 OFDM subcarrier snapshot per position
(no time/velocity axis). The paper's two structured axes (OFDM subcarrier,
OFDM symbol/time) become (subcarrier, TX antenna) here:

```
Nc <- subcarrier / frequency   (delay-stationary  -> Toeplitz)
Nt <- antenna    / ULA element (angle-stationary  -> Toeplitz)
C = kron(C_ant, C_freq)   <->   the paper's kron(C_time, C_freq)
```

This folder is self-contained: every path in the kept scripts is relative
to `Share/`, and the train/test channel datasets are already included in
`Data/`, so you can reproduce the experiments with **no GPU / no Sionna
install** (see Quick start). Ray-tracing the datasets from scratch is only
needed if you want to regenerate or extend them (see "Regenerating the
channel data").

## Layout

```
Share/
  requirements.txt
  pipeline/
    config.py                              shared constants (M, F, subcarrier spacing)
    scene_geometry.py                       POSTECH scene loader / building-footprint helpers
    generate_channels_subregion100.py       ray-tracing core (imported by *_marray.py below);
                                             also runnable standalone to regenerate the
                                             M=16 dense-grid training set
    generate_channels_subregion100_marray.py  ray-traces train/test_M{m}_F64_subregion100.npz
                                             for an arbitrary TX array size --m
    GMM_CE/
      run_subregion_experiments.py          main entry point: fits & evaluates all GMM-CE
                                             estimators, writes NMSE plots/JSON
      gmmce/                                the estimator library (see gmmce/gmmce_README.txt
                                             for the internal module dependency layering)
      results_subregion_M16/                output of --m 16 (quick scale)
      results_subregion_M4[...]/             outputs of --m 4 under various --scale /
                                             --comb-spacing sweeps (see table below)
  Data/
    positions_subregion_grid_0.25m_100x100.npz   dense grid (bbox + xy positions)
    train_M4_F64_subregion100.npz, test_M4_F64_subregion100.npz
    train_M16_F64_subregion100.npz, test_M16_F64_subregion100.npz
  POSTECH_scene/                            Sionna RT scene (POSTECH.xml + meshes),
                                             needed only to regenerate the Data/*.npz above
  legacy/                                   everything from the original project that is
                                             NOT part of this experiment (see legacy/README.md)
```

Keep `pipeline/`, `pipeline/GMM_CE/`, `Data/` and `POSTECH_scene/` in this
relative arrangement -- every script locates its inputs/outputs relative to
its own file location or the documented working directory, never via an
absolute path.

## 0. Dependencies

```bash
pip install -r requirements.txt
```

Running `run_subregion_experiments.py` on the included datasets only needs
`numpy`/`scipy`/`matplotlib` (pure CPU, no GPU). `sionna-rt`/`tensorflow`
are only imported by the two `generate_channels_subregion100*.py` scripts
(re-ray-tracing the scene); `torch` is only imported by `--gpu`
(`gmmce/gpu_gmm.py`).

If you do run the ray tracer on a shared GPU server: `pipeline/scene_geometry.py`
defaults `CUDA_VISIBLE_DEVICES` to `2` (`os.environ.setdefault`, so it only
applies if you haven't already set one yourself) -- that was this project's
lab machine, not a requirement. Override it for your own machine, e.g.:
```bash
export CUDA_VISIBLE_DEVICES=0   # or whatever's free/available on your box
```

## Quick start (no GPU needed -- uses the included datasets)

```bash
cd pipeline/GMM_CE
python3 run_subregion_experiments.py --scale quick               # 1x4 TX, full pilot, K=8 N=3k  (~2.5 min)
python3 run_subregion_experiments.py --scale quick --m 16         # 1x16 TX, D=512
python3 run_subregion_experiments.py --scale quick --comb-spacing 4   # 1x4 TX, comb-4 pilot
```
Each run writes into `results_subregion_M{m}[_comb{s}][_{scale}]/`
(the `quick` scale has no `_{scale}` suffix):

| file | paper analogue | x-axis |
|---|---|---|
| `nmse_vs_snr.png` / `.json`        | Fig. 3  | SNR [dB] |
| `nmse_vs_training.png` / `.json`   | Fig. 4a | Training data size |
| `nmse_vs_components.png` / `.json` | Fig. 4b | GMM components K |

Redraw plots from existing JSON without recomputing:
```bash
python3 run_subregion_experiments.py --replot --m 4 --comb-spacing 4
```

Larger scales (heavier, CPU or `--gpu`):
```bash
python3 run_subregion_experiments.py --scale k16n10k               # K=16, N=10k  (~30 min, CPU)
python3 run_subregion_experiments.py --scale k16n40k --gpu         # K=16, N=40k  (~14 min, GPU/torch)
python3 run_subregion_experiments.py --scale paper50 --comb-spacing 8 --gpu  # paper's Sec. V settings, ~1.5h/config
```
Scales: `quick` (K=8,N=3k), `k16n10k` (K=16,N=10k), `k16n40k` (K=16,N=40k),
`demo` (K=16,N=5k), `paper` (K=128,N=1e5,n_iter=25), `paper50` (K=128,
N=1e5, n_iter=50 -- the paper's Sec. V settings; needs a >=100k train pool).
`--gpu` (`gmmce/gpu_gmm.py`) batches the EM E/M-steps (and the CME eval)
over K with torch. Mixed precision: `GMM full`/`kron` run complex128
(fp32 loses ~1.3 dB on their near-singular dense-cov LMMSE); `b-toep`/
`b-circ` run complex64 -- set `GMMCE_GPU_STRUCT_DTYPE=complex128` to force
fp64 there too. GPU results match CPU within EM-convergence noise
(covariance rel err ~1e-9, CME rel err ~1e-14).

## Regenerating the channel data (optional, needs sionna-rt + the POSTECH scene)

`Data/train_M{4,16}_F64_subregion100.npz` and
`test_M{4,16}_F64_subregion100.npz` are already included, so this step is
only needed if you want to change M, F, or the train/test split:

```bash
cd pipeline
CUDA_VISIBLE_DEVICES=0 python3 -u generate_channels_subregion100_marray.py --m 4  --n-train 45000 --n-test 8000
CUDA_VISIBLE_DEVICES=0 python3 -u generate_channels_subregion100_marray.py --m 16 --n-train 45000 --n-test 8000
```
Physics: fc 3.5 GHz, SCS 240 kHz, F=64 subcarriers, TX ULA at
`(-70, -25, 15)`, diffuse reflection on, `max_depth=5`,
`samples_per_src=1e6`, seed 42. `--m 16`'s test pool defines the held-out
positions reused (via `test_M16_F64_subregion100.npz`) for every other
`--m`, so results stay comparable across array sizes.

## Results summary

**`--scale quick`, 1x4 TX (K=8, N_train=3000, D=256):**

*Full pilot:* `GMM full` ~= `GMM 2x1D` ~= `GMM b-toep` ~= `GMM kron` track
together across the whole SNR range (`b-toep` slightly best at high SNR --
Toeplitz regularisation beats the `K*D^2`-parameter `full` fit at
N=3000); `b-circ` / `2x1D-circ` trail by a roughly fixed factor; the genie
`PDP+DS` baseline sits with the `b-circ` group. In NMSE-vs-training,
`b-toep` is the most data-efficient (~2e-2 from N=375); the
"`full` collapses at small N" Fig. 4a effect is starkest at `--m 16`
(`full` = 0.28 @ N=150 vs structured ~0.02).

*Comb pilot (`--comb-spacing 4` / `8`):* the picture separates the way the
paper's Fig. 3 does -- **`b-circ`, `2x1D-circ` and the genie `PDP+DS`
baselines hit a hard NMSE floor** (comb-4: ~2.4e-2 / ~2.7e-2 at 30 dB)
because a DFT/circulant delay model aliases when pilots are subsampled,
while **`GMM full` / `b-toep` / `2x1D` / `kron` keep descending**
(`b-toep` best: 6.2e-4 at 30 dB, comb-4). The GMM-with-Toeplitz-prior
advantage over the PDP/DS genie -- one of the paper's headline claims --
is only visible once the pilot grid is sparse. Gap widens further at comb-8.

**`--scale k16n10k` (K=16, N=10 000):** same ordering as `quick`, but
`GMM full` closes on `GMM b-toep` as K/N grow -- full-pilot 30 dB `b-toep`
1.5e-4 vs `full` 1.7e-4 (a ~0.5 dB gap, vs ~2 dB at `quick` K=8/N=3k).

**`--scale paper50` (K=128, N=100 000, n_iter=50, GPU)** -- the paper's
Sec. V settings: **`GMM full` is now best at every SNR**, and the whole
ordering reproduces Fesl Fig. 3/4: `full` < `b-toep` < `2x1D` ~
`2x1D-toep` < `kron` < `b-circ` ~ `2x1D-circ` < genie `PDP+DS` (full
pilot, 30 dB: 9.3e-5 / 1.4e-4 / 1.4e-4 / 1.9e-4 / 5.7e-4 / 7.5-9.4e-4).
comb-8: `b-circ`/`2x1D-circ`/genie still floor (aliasing), `full` best for
SNR >= ~5 dB, `b-toep` best only at -10/0 dB.

### Why `GMM b-toep` beats `GMM full` at small K/N (the paper -- and paper50 -- have `full` best)

The paper's Fig. 3/4 ordering (`full` >= `b-toep` >= `kron` ~ `2x1D` >
`b-circ` > genie) is measured with a **lattice pilot `Np = 50/336`,
`K = 128`, `N_train = 1e5`** on a **diffuse** synthetic/QuaDRiGa channel
(200 continuous-delay paths, exponential PDP). Our quick runs differ on
the channel *and* the operating point, and both push `b-toep` above `full`:

1. **The channel.** With the paper's own synthetic channel and everything
   else identical to our run (full pilot, K sweep, N=3000), `GMM full` is
   ~0.4 dB *better* than `b-toep` at every K 1..32 -- the paper's ordering.
   On the ray-traced POSTECH channel the same sweep flips: `full` ==
   `b-toep` at K=1, then `b-toep` is ~2 dB better for every K >= 4. The
   POSTECH channel is **specular** -- a few sharp, off-grid,
   strongly position-dependent paths (test-cov effective rank ~25/256 vs
   ~8/336 for the synthetic). A GMM component's covariance is therefore
   peaky and, estimated *unconstrained* from only `N/K` ~ 375 samples for
   a 65 536-entry matrix, high-variance. A ULA + per-position-WSSUS
   channel's covariance is exactly block-Toeplitz, so `b-toep` projects
   that noisy estimate onto the right ~10^3-dim subspace -- near-zero
   bias, large variance reduction. The diffuse synthetic channel's
   component covariances are smooth enough that `full` estimates them
   cleanly from the same `N/K`, so its zero bias wins -- as in the paper.
2. **Operating point.** `K=8, N=3000` (quick) vs the paper's `K=128,
   N=1e5`. The `full`-vs-`b-toep` gap on the POSTECH data *shrinks* as K
   grows (2.1 dB -> 1.6 dB over K=8->32), and `full` keeps closing it with
   the paper's K/N (`--scale paper`).
3. **Full pilot.** With `A = I` there is no interpolation, so covariance
   *bias* barely matters and the whole field compresses; the estimator
   that denoises with the lowest-variance covariance (`b-toep`) wins by a
   modest constant. The paper's sparse lattice pilot is what makes
   modeling bias (and `b-circ`'s aliasing floor, and the genie's floor)
   dominate -- reproduced by the `--comb-spacing 4/8` runs above.

Not a bug: `gmmce/` faithfully implements the paper's Eqs. (2)-(7), and on
the paper's own channel model this pipeline reproduces the paper's
estimator ordering.

### Fixes made to `GMM_CE/gmmce/` (behaviour-preserving, verified)

1. **`n_iter_toep` in `train_all`** (default `max(10*n_iter, 120)`): the
   Barton & Fuhrmann Toeplitz M-step (paper Eq. 6) is one gradient-ascent
   step on the circulant-embedding spectrum per EM iteration -- its
   log-likelihood rises ~linearly and needs ~10x more iterations than the
   closed-form sample-covariance M-step of `full`/`kron`/`circ` to
   converge, more so on a sharply-structured ray-traced covariance far
   from the flat-spectrum init. Giving only the Toeplitz variants ~120
   iterations removes an NMSE floor that used to stop `b-toep` improving
   past ~5 dB SNR (`fit_toeplitz_gmm`'s own log-likelihood `tol` still
   early-stops it once converged).
2. **`fit_circulant_gmm`** rebuilds each `C_k = F^H diag(c_k) F` from the
   diagonal instead of a dense `einsum` with a naive `O(K D^4)` path
   (unusable past `D~256`).
3. **`covariances_from_pdp_ds`** einsums get `optimize=True`.

## `legacy/`

Everything from the original multi-experiment project that this
experiment doesn't need: mobility channel prediction, Doppler-cube
generation, moving-pedestrian trajectory NMSE, the dense-map (non-array)
EM sweep, the generic synthetic-channel Fig.2/3/4 reproduction
(`gmmce/experiments.py` + `run_experiments.py`), paper-reference text/image
dumps, and the previous `README.md`/`guideline.md`. See
[legacy/README.md](legacy/README.md) for what's there and why it was set
aside.
