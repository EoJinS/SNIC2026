# GMM-CE on subregion100

Structured-covariance GMM channel estimation (Fesl et al., Asilomar 2022:
`full` / `b-toep` (block-Toeplitz) / `b-circ` (block-circulant) / `kron` /
`2x1D` / `2x1D-toep` / `2x1D-circ`) evaluated on **position-only** channels
ray-traced with Sionna RT over a dense 0.25 m grid of the POSTECH campus
("`subregion100`": `x in [-80,20]`, `y in [-60,40]`).

There are THREE channel datasets / experiment drivers sharing the same
estimator library (`gmmce/pipeline.py` et al., which only ever sees a
generic `(Nc, Nt)` pair -- see `gmmce/pilots.py`):

1. **MISO space-frequency** (`run_subregion_experiments.py`, the original):
   a 1xM TX ULA x F=64 OFDM subcarrier snapshot per position (no
   time/velocity axis). The paper's two structured axes (OFDM subcarrier,
   OFDM symbol/time) become (subcarrier, TX antenna) here:
   ```
   Nc <- subcarrier / frequency   (delay-stationary  -> Toeplitz)
   Nt <- antenna    / ULA element (angle-stationary  -> Toeplitz)
   C = kron(C_ant, C_freq)   <->   the paper's kron(C_time, C_freq)
   ```
2. **SISO time-frequency** (`run_subregion_tf_experiments.py`, added
   2026-09-11): Fesl's OWN axes -- a SISO channel, F=64 subcarriers x
   S=14 OFDM symbols per position, UE speed 10 m/s (random heading), so
   the time axis is a genuine OFDM-symbol/Doppler axis, not an antenna
   proxy:
   ```
   Nc <- subcarrier / frequency   (delay-stationary   -> Toeplitz)
   Nt <- OFDM symbol / time       (Doppler-stationary  -> Toeplitz)
   C = kron(C_time, C_freq)   <->   exactly the paper's own ordering
   ```
   See "SISO time-frequency experiment" below.
3. **1x4 ULA velocity sweep** (same two driver scripts above, `--data-tag
   ulatf`, added 2026-09-16/17): space-freq (antenna x freq) AND time-freq
   (freq x OFDM symbol) as two 2-D slices of the SAME mobile 1x4-TX-ULA
   system (`generate_channels_subregion100_ula_tf.py`), swept over UE
   speed 0/3/6/10 m/s and full-pilot/comb-8, with `GMM b-toep` excluded.
   See "1x4 ULA velocity sweep" below.

This folder is self-contained: every path in the kept scripts is relative
to `Share/`, and the train/test channel datasets are already included in
`Data/`, so you can reproduce the MISO experiments with **no GPU / no
Sionna install** (see Quick start). Ray-tracing the datasets from scratch
is only needed if you want to regenerate or extend them (see "Regenerating
the channel data"); the SISO time-frequency data pool is large (paper50
scale needs >=100k train + >=9k test positions) and is regenerated the
same way (see below), not shipped by default.

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
    generate_channels_subregion100_siso_tf.py  ray-traces
                                             train/test_siso_S{s}_F{f}_v{v}_subregion100.npz
                                             (SISO, F subcarriers x S OFDM symbols, Fesl's axes)
    generate_channels_subregion100_ula_tf.py  ray-traces train/test_ula{m}_S{s}_F{f}_v{v}_subregion100.npz
                                             (1xM TX ULA x F subcarriers x S OFDM symbols x
                                             velocity -- ONE mobile-ULA system; see "1x4 ULA
                                             velocity sweep" below for why this exists
                                             alongside the two scripts above)
    derive_ula_tf_slices.py                 slices the ula_tf cube into the two 2-D pools
                                             below (*_ulatf.npz)
    GMM_CE/
      run_subregion_experiments.py          MISO space-freq entry point: fits & evaluates all
                                             GMM-CE estimators, writes NMSE plots/JSON
      run_subregion_tf_experiments.py       SISO time-freq entry point (same estimators/plots)
      run_weichselberger_addon.py            fits ONLY the GMM Weichselberger estimator and
                                             merges it into existing results (see below)
      gmmce/                                the estimator library (see gmmce/gmmce_README.txt
                                             for the internal module dependency layering);
                                             subregion_data.py = MISO adapter,
                                             subregion_tf_data.py = SISO time-freq adapter
      results_subregion_M16/                output of --m 16 (quick scale)
      results_subregion_M4[...]/             outputs of --m 4 under various --scale /
                                             --comb-spacing sweeps (see table below)
      results_subregion_tf[...]/            outputs of run_subregion_tf_experiments.py
      results_subregion_M4_ulatf_paper50/   1x4 ULA space-freq slice, full pilot (see below)
      results_subregion_tf_v{0,3,6,10}_ulatf_paper50/  1x4 ULA time-freq slice, full pilot, per velocity
      results_subregion_M4_ulatf_comb8_paper50/         same, comb-spacing-8 pilot
      results_subregion_tf_v{0,3,6,10}_ulatf_comb8_paper50/  same, comb-spacing-8 pilot, per velocity
  Data/
    positions_subregion_grid_0.25m_100x100.npz   dense grid (bbox + xy positions)
    train_M4_F64_subregion100.npz, test_M4_F64_subregion100.npz
    train_M16_F64_subregion100.npz, test_M16_F64_subregion100.npz
    train_siso_S14_F64_v10_subregion100.npz, test_siso_S14_F64_v10_subregion100.npz
    train/test_ula4_S14_F64_v{0,3,6,10}_subregion100.npz          raw 1x4 ULA mobility cubes
    train/test_M4_F64_subregion100_ulatf.npz                      space-freq slice (from v=0 cube)
    train/test_siso_S14_F64_v{0,3,6,10}_subregion100_ulatf.npz    time-freq slice, per velocity
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

## SISO time-frequency experiment (Fesl's own axes)

`generate_channels_subregion100_siso_tf.py` ray-traces a SISO channel
(single TX antenna) laid out on the paper's own axes -- F=64 OFDM
subcarriers x S=14 OFDM symbols (one NR slot) per position -- instead of
the MISO (subcarrier, TX-antenna) proxy above. One `PathSolver()` call per
position gives all 14 symbol snapshots via Sionna's analytic Doppler
(`Receiver(velocity=...)` + `Paths.cfr(sampling_frequency=1/71.4us,
num_time_steps=14)`); UE speed defaults to 10 m/s (random heading per
position, no static UE-position axis is added on top) -- chosen so the
14-symbol / ~1 ms slot decorrelates by ~tens of degrees of Doppler phase
(Jakes/Clarke spectrum, verified: |correlation| decays smoothly from 1.0
to ~0.9 across the slot -- genuinely time-selective, not degenerate). See
the script's docstring for the full derivation. Same POSTECH scene
physics as every other script here (fc 3.5 GHz, TX `(-70,-25,15)`, diffuse
on, `max_depth=5`, `samples_per_src=1e6`, seed 42).

### 1. Regenerate the data (needed once -- not shipped, paper50 scale needs a large pool)

```bash
cd pipeline
CUDA_VISIBLE_DEVICES=2 python3 -u generate_channels_subregion100_siso_tf.py --n-train 110000 --n-test 12000
```
~13 min on the lab GPU (`scene_geometry.py` pins `CUDA_VISIBLE_DEVICES=2`
by default; override to whatever's free on your box). Writes
`Data/{train,test}_siso_S14_F64_v10_subregion100.npz`. `--speed`/
`--n-symbols`/`--f`/`--symbol-dt`/`--heading` let you change the
setup (filenames auto-encode `S`/`F`/`v`; `run_subregion_tf_experiments.py`
must be given matching `--n-symbols`/`--f`/`--speed` if you deviate from
the defaults).

### 2. Run the experiments

```bash
cd pipeline/GMM_CE
# smoke test (CPU, ~1 min on GPU / a few min CPU) -- needs a small data pool, e.g. --n-train 4000 --n-test 1200 above
python3 run_subregion_tf_experiments.py --scale quick --no-gpu

# task setting: K=128, N_train=1e5, EM iterations=50, GPU (paper50 is the default --scale here)
python3 run_subregion_tf_experiments.py                       # full pilot
python3 run_subregion_tf_experiments.py --comb-spacing 8      # freq-comb-8 pilot (Np=8*14/896)

# redraw plots from existing JSON, no recompute
python3 run_subregion_tf_experiments.py --replot --comb-spacing 8
```
`--gpu` is **ON by default** for this script (pass `--no-gpu` to force
CPU). Each `paper50` config writes into
`results_subregion_tf[_comb{s}]_paper50/`:

**Known GPU precision issue (found 2026-09-11) -- pass `GMMCE_GPU_STRUCT_DTYPE=complex128`
if you need trustworthy `GMM b-toep` numbers, especially under comb
pilots.** `gmmce/gpu_gmm.py` runs `GMM b-toep`/`b-circ` in `complex64` by
default (see "Regenerating the channel data" above for why that's fine at
the MISO scale). At this SISO time-freq scale (D=896, K=128),
`complex64` is numerically insufficient for `b-toep`'s CME step: under
the FULL pilot it's not catastrophic (a consistent ~1.4-9x worse than
`GMM full` instead of tracking/beating it as in the paper/MISO
precedent); under `--comb-spacing 8` (Np=112, a heavily ill-posed
Np<<D problem) it's a hard break -- `GMM b-toep`'s NMSE plateaus flat
around 0.6-0.8 regardless of SNR, training size, or K (isolated via a
from-scratch K-sweep and a same-model full-vs-comb8 A/B comparison --
ruled out under-convergence, K-scaling, and N-scaling individually; only
the JOINT 2-D `b-toep` estimator is affected, not `2x1D-toep`'s two
independent 1-D Toeplitz fits). `GMMCE_GPU_STRUCT_DTYPE=complex128`
fixes both completely (see Results below) at a real cost: **fitting is
much slower** -- ~5x at a small diagnostic scale (K=128 fit: 325s
complex64 -> 1675s complex128 at N=5000), and at the full `paper50`
K=128/N=100000 scale the `nmse_vs_snr` sweep alone (one `train_all` +
5 evals) took ~5-6h per config instead of ~87 min, with the full
3-plot config's first `train_all` still running past 3h with no end in
sight before being stopped. **Practical recommendation at this scale:**
confirm the fix with just the SNR sweep, not the full 3-plot config:
```bash
GMMCE_GPU_STRUCT_DTYPE=complex128 python3 run_subregion_tf_experiments.py \
    --skip fig4a fig4b --outdir results_subregion_tf_paper50_fp64
GMMCE_GPU_STRUCT_DTYPE=complex128 python3 run_subregion_tf_experiments.py \
    --comb-spacing 8 --skip fig4a fig4b --outdir results_subregion_tf_comb8_paper50_fp64
```
Only re-run the full `nmse_vs_training`/`nmse_vs_components` sweeps at
`complex128` if you specifically need trustworthy `GMM b-toep` numbers
there and can afford the multi-day runtime (or drop to a smaller
`--scale` first).

| file | paper analogue | x-axis |
|---|---|---|
| `nmse_vs_snr.png` / `.json`        | Fig. 3  | SNR [dB] |
| `nmse_vs_training.png` / `.json`   | Fig. 4a | Training data size |
| `nmse_vs_components.png` / `.json` | Fig. 4b | GMM components K |

### Results (SISO time-freq, K=128, N_train=100000, n_iter=50, GPU, `GMMCE_GPU_STRUCT_DTYPE=complex64` default)

Both configs completed (`results_subregion_tf_paper50/`, ~6h59m;
`results_subregion_tf_comb8_paper50/`, ~6h35m -- this SISO D=896 scale is
far heavier than the MISO D<=1024 precedent, mostly from the CME
evaluation, not the EM fit).

**Full pilot** (Np=896/896) @30dB: `full` 3.1e-5 < `kron` 3.8e-5 <
`2x1D` 5.7e-5 < `2x1D-toep` 1.1e-4 < `b-circ` 2.0e-4 < `b-toep` 2.7e-4 <
`PDP+DS kron` 3.8e-4 < `PDP+DS 2x1D` 6.3e-4. `full`/`kron` track together
and lead throughout -10..30 dB; `b-toep` sits a consistent ~9x behind
`full` (see the precision note above) rather than beating it as in the
paper/MISO precedent.

**Comb-8** (Np=112/896) @30dB: `full` 3.4e-3, `kron` 6.6e-3, `2x1D` 2.8e-3,
`2x1D-toep` 3.3e-3 all close together and clearly best; `b-circ`/
`2x1D-circ`/`PDP+DS` hit the expected aliasing floor (~0.02-0.08, DFT/
circulant delay model breaks under subsampled pilots -- reproduces the
paper's Fig. 3 story); **`b-toep` (only) is broken** -- flat ~0.62-0.76
across ALL SNR/training-size/K points (sometimes >1.0). Note `2x1D-toep`
(two INDEPENDENT 1-D Toeplitz fits, cascaded) stays fine throughout --
only the single JOINT 2-D block-Toeplitz `b-toep` estimator hits the
precision bug documented above, not every Toeplitz-structured estimator. Ignore
`GMM b-toep`'s comb-8 numbers from this run and use the
`GMMCE_GPU_STRUCT_DTYPE=complex128` rerun instead (below).

### Results with the precision fix (`GMMCE_GPU_STRUCT_DTYPE=complex128`)

`nmse_vs_snr` reproduced for both pilot configs (`results_subregion_tf_paper50_fp64/`,
`results_subregion_tf_comb8_paper50_fp64/`) to confirm the fix -- **not**
`nmse_vs_training`/`nmse_vs_components`, which would need re-running every
one of the 8 `train_all` calls per config at the ~5x complex128 slowdown
(each config's `nmse_vs_snr` alone -- one `train_all` + 5 SNR evals --
already took ~5-6h at this D=896, K=128, N=100000, Np up to 896 scale,
vs ~87 min for the equivalent complex64 call; the full 3-plot config was
running past 3h on its first `train_all` alone with no end in sight, so
it was stopped in favour of just confirming Fig. 3).

**Full pilot**, complex64 -> complex128 (`GMM b-toep`, -10..30 dB):
0.220->0.044, 0.056->0.0085, 0.0134->0.0016, 0.0021->0.00027,
2.75e-4->4.33e-5 -- now tracks `full`/`kron` within ~1.4x at every SNR
(was up to 9x off), matching the paper/MISO precedent.

**Comb-8**, complex64 -> complex128 (`GMM b-toep`, -10..30 dB):
0.762->0.197, 0.615->0.041, 0.650->0.0101, 0.704->0.0043,
0.720->0.0032 -- from a flat, broken ~0.6-0.76 to tracking `full` almost
exactly, and slightly BEATING it at 20/30 dB (`b-toep` 0.00427/0.00322
vs `full`'s 0.00432/0.00336) -- now correctly reproduces the paper's headline
"structured Toeplitz prior beats naive full-pilot inversion under sparse
pilots" result. `b-circ`/`2x1D-circ`/`PDP+DS` are unaffected by the
dtype (still floor from aliasing, as expected -- that's the DFT/circulant
delay-model mismatch, not a precision bug).

**Takeaway**: `GMMCE_GPU_STRUCT_DTYPE=complex128` is required for
trustworthy `GMM b-toep` numbers at this K=128/D=896 SISO scale,
especially under sparse pilots; complex64 is fine for every other
estimator at every scale tried in this file, and fine for `b-toep` too
at the smaller MISO scales (D<=1024) documented earlier in this file.

## 1x4 ULA velocity sweep: space-freq + time-freq as two slices of ONE system

Added 2026-09-16/17 per `guideline.md`: besides the MISO space-freq experiment
above, look at the time-frequency axis too as UE velocity varies, for the
SAME 1x4 TX ULA -- i.e. space-freq (antenna x freq) and time-freq (freq x
OFDM symbol) should be two 2-D slices of one mobile 1x4-ULA system, not two
different systems that happen to share physics constants. (An earlier pass
mistakenly paired the existing static M4 space-freq data with the pre-existing
*standalone-single-antenna* SISO time-freq script -- physically distinct from
"one element of the same 4-antenna array" because of the sub-wavelength
array-element offset, which changes multipath phase. Fixed as below.)

**New data pipeline** (`pipeline/`):
1. `generate_channels_subregion100_ula_tf.py --speed {0,3,6,10}` ray-traces
   one `(M=4 antenna, S=14 OFDM symbol, F=64 subcarrier)` cube per position
   per velocity -- TX array identical to `generate_channels_subregion100_marray.py`
   (`PlanarArray(1,4)`, 0.5-spacing, iso, V-pol), Doppler/time mechanism
   identical to `generate_channels_subregion100_siso_tf.py` (`Receiver(velocity=...)`
   + analytic-Doppler `cfr(sampling_frequency=1/71.4us, num_time_steps=14)`).
   Same scene physics as everywhere else (fc 3.5 GHz, TX (-70,-25,15), diffuse
   on, max_depth 5, samples_per_src 1e6, seed 42). `--speed 0` gives a
   no-Doppler baseline (all 14 symbols numerically identical, verified). Same
   train/test position pools (110000/12000) across every velocity (only
   velocity changes between runs). Writes `Data/{train,test}_ula4_S14_F64_v{v}_subregion100.npz`
   (~3.1 GB / ~345 MB each -- ray-tracing the 4x-bigger CFR tensor needs a
   ~4x smaller drjit chunk size than the SISO script, ~150 pos/s, ~12-13 min
   train + ~1.5 min test per velocity).
2. `derive_ula_tf_slices.py` slices each cube (the GMM-CE estimator library,
   `gmmce/pipeline.py`, only ever sees a generic 2-D `(Nc, Nt)` pair, so a
   true 3-D antenna/freq/time estimator isn't supported -- this is why "two
   2-D graphs" rather than one 3-D one):
   - **space-freq** (antenna x freq): the OFDM-symbol-0 slice of the
     `--speed 0` cube -> `Data/{train,test}_M4_F64_subregion100_ulatf.npz`.
     Velocity-invariant by construction (0 Doppler phase at t=0) --
     verified numerically: the v=0 and v=3 cubes' symbol-0 slices agree to
     ~5e-7 (float32 noise) on a ~3e-4 signal. Generated once.
   - **time-freq** (freq x OFDM symbol): TX antenna index 0's slice of each
     velocity's cube -> `Data/{train,test}_siso_S{14}_F64_v{v}_subregion100_ulatf.npz`
     (same file layout `gmmce/subregion_tf_data.py` already expects, so no
     adapter code was needed -- just a `--data-tag` to point at it). One per
     velocity (0/3/6/10 m/s); v=0's Doppler-free antenna-0 slice differs
     from v=0's own symbol-0-across-antennas slice above (different fixed
     index into the same cube), so both are legitimate, independent 2-D
     views of the identical raw data.

**How the channel data actually trains the EM fit (and where velocity fits
in):** velocity is NOT an input to the EM algorithm -- it only decides
*which data pool gets loaded*. Once loaded, the fitter never sees velocity,
position, or antenna/OFDM-symbol index; it only ever sees plain complex
channel VALUES. Concretely, per `--speed v` run:

1. `train_siso_S14_F64_v{v}_subregion100_ulatf.npz["h"]` holds `N` ray-traced
   complex channel realizations as one flat `(N, S*F)` array -- no velocity
   label is stored per-sample. Velocity only shaped these VALUES upstream,
   during ray-tracing (higher speed -> the 14-symbol snapshot decorrelates
   faster across the time axis), not something passed into the fit.
2. `SubregionChannelsTF.__init__` (`gmmce/subregion_tf_data.py`) reshapes
   that flat array to `(N, Nc=64, Nt=14)` and rescales all `N` samples by
   one real scalar so `E[||vec(H)||^2] = Nc*Nt` (`SubregionChannels` does
   the analogous `(antenna, freq)` reshape for space-freq). `.train(n)`
   then slices out a random `n`-sized subset of these clean `(Nc,Nt)`
   matrices -- e.g. `Htr = data.train(100000)` -> a `(100000, 64, 14)`
   complex128 array. This is a **fixed, disjoint** subset for train vs. the
   `.calib(n)`/`.test(n)` pools held out for calibration/evaluation --
   still no pilots, no noise, no SNR at this point.
3. This `Htr` is exactly what's handed to `train_all(Htr, grid, cfg, K=128,
   ..., which=FIT_VARIANTS)` (`gmmce/pipeline.py`). Inside it, `Xtr =
   vec(H_train)` flattens each `(Nc,Nt)` matrix to one length-`D=Nc*Nt`
   (=896 here) complex vector, so `Xtr` is `(100000, 896)` -- THAT is the
   literal input to the EM fitters:
   - `fit_full_gmm(Xtr, K=128, n_iter=50)`: unconstrained K=128-component
     complex-Gaussian-mixture EM (per-component sample mean + covariance
     M-step) on the 896-dim vectors.
   - `fit_toeplitz_gmm(Xtr, K=128, dims=[Nt,Nc], n_iter=n_iter_toep)`
     (skipped by `--no-btoep`) / `fit_circulant_gmm(...)`: same E-step,
     but the M-step constrains each component's covariance to be
     block-Toeplitz / block-circulant over the `(Nt,Nc)` structure (see
     "Fixes made to `GMM_CE/gmmce/`" above for the Toeplitz M-step detail).
   - For `kron`/`2x1D`/`2x1D-toep`/`2x1D-circ`, `train_all` additionally
     slices `H_train` into its `Nt`-length columns (fixed frequency) and
     `Nc`-length rows (fixed time), fits separate 1-D GMMs on each marginal
     (same `fit_*_gmm` functions, just `D=Nt` or `D=Nc`), then combines
     them (`combine_kronecker` / `Cascade2x1D`).
4. The pilot grid `A` (full or `--comb-spacing 8`) and the noise variance
   `sigma2` (from the SNR sweep) never appear during this fit -- they're
   only used LATER, at evaluation (`evaluate_joint`/`evaluate_cascade`),
   when the already-trained GMM's conditional-mean estimator reconstructs
   held-out `Hte` samples from noisy pilots `y = A h + n`.

So "training per velocity" means 4 fully independent runs: each `--speed v`
loads a different data pool and calls `train_all` fresh, producing its own
K=128-component GMM. There is no single model conditioned on velocity, and
no velocity value ever enters the EM math -- the only effect velocity has
is upstream, on which channel realizations end up in `Htr`/`Hte`.

**Code changes** (`pipeline/GMM_CE/`, both `run_subregion_experiments.py`
and `run_subregion_tf_experiments.py`):
- `--data-tag TAG` -> loads `train/test_..._subregion100_TAG.npz` instead of
  the plain filename (`gmmce/subregion_data.py`, `gmmce/subregion_tf_data.py`).
  The pre-existing, untagged datasets (static M4, standalone SISO v10) are
  untouched.
- `--no-btoep` -> `GMM b-toep` is neither fit (`train_all`'s `which=`) nor
  plotted/legended. Opt-in (default off) so other experiments can still use
  it; used here per guideline condition 1 (its Barton-Fuhrmann M-step needs
  ~10x more EM iterations than every other variant -- a large chunk of the
  runtime with little payoff for this sweep).
- `--scale paper50` no longer appends `[paper50: K=..., N=...]` to the plot
  title (still appended for every other `--scale`).
- `run_subregion_tf_experiments.py`'s title/outdir now include `v{speed}m/s`
  so the 4 velocity runs are distinguishable; its default `--outdir` gained
  a `_v{speed}` prefix (existing `results_subregion_tf_paper50/` etc. from
  before this change are untouched -- only new default-outdir runs use the
  new naming).

**Reproduce** (`cd pipeline/GMM_CE`, needs `Data/*_ulatf.npz` from steps 1-2
above; GPU used per guideline condition 4):
```bash
python3 run_subregion_experiments.py --m 4 --scale paper50 --no-btoep --data-tag ulatf --gpu
# -> results_subregion_M4_ulatf_paper50/
for v in 0 3 6 10; do
  python3 run_subregion_tf_experiments.py --speed $v --scale paper50 --no-btoep --data-tag ulatf --gpu
done
# -> results_subregion_tf_v{0,3,6,10}_ulatf_paper50/
```
Each writes the usual 3 plots (`nmse_vs_snr.png`, `nmse_vs_training.png`,
`nmse_vs_components.png`) + matching `.json`. Wall time (GPU, `--no-btoep`,
full pilot): space-freq (D=256) ~71 min; time-freq (D=896) ~3.4-4.6h per
velocity (v=0 fastest, v=10 slowest -- higher Doppler spread makes the
channel mildly harder for every estimator, see below).

**Results** (K=128, N_train=1e5, n_iter=50, full pilot, 30 dB SNR unless noted):

*Space-freq (1x4 ULA, antenna x freq, D=256):* `full` 1.4e-4 < `2x1D` 2.1e-4
< `2x1D-toep` 2.4e-4 < `kron` 3.1e-4 < `2x1D-circ` 4.8e-4 < `b-circ` 6.0e-4 <
`PDP+DS 2x1D` 7.7e-4 < `PDP+DS kron` 1.0e-3 -- same ordering as the
paper50 MISO result earlier in this file (`full` best throughout -10..30 dB).

*Time-freq (freq x OFDM symbol, D=896), `full`/`kron` NMSE @30dB by velocity:*
v=0: 2.1e-5 / 2.4e-5; v=3: 2.8e-5 / 3.3e-5; v=6: 3.2e-5 / 4.0e-5; v=10:
3.6e-5 / 4.4e-5 -- NMSE degrades smoothly and monotonically with speed for
every estimator (full pilot means no time-axis interpolation is needed, so
this is the GMM prior finding the channel's structure mildly harder to
exploit as Doppler spread grows, not an aliasing/interpolation effect).
`full` < `kron` < `2x1D` < `2x1D-toep` < `b-circ` < `2x1D-circ` < `PDP+DS`
pair at every velocity (`2x1D-circ` and `PDP+DS kron` are within ~1% of each
other at v=0, close enough to swap order on EM noise) -- same ordering as
the space-freq result and the original SISO v=10 paper50 result earlier in
this file.

The raw `ula4_S14_F64_v*` cubes (~14 GB total across 4 velocities) are kept
under `Data/` for provenance/reuse; delete them if disk space is needed --
only the much smaller `*_ulatf.npz` slices are read by the experiment
scripts above.

### Comb-spacing-8 sweep (2026-09-17)

Same 5 configs, same data, `--comb-spacing 8` added (frequency-sparse
pilot: space-freq Np=32/256 -- Npc=8 of 64 subcarriers x all 4 antennas;
time-freq Np=112/896 -- Npc=8 of 64 subcarriers x all 14 symbols):
```bash
python3 run_subregion_experiments.py --m 4 --scale paper50 --no-btoep --data-tag ulatf --comb-spacing 8 --gpu
# -> results_subregion_M4_ulatf_comb8_paper50/
for v in 0 3 6 10; do
  python3 run_subregion_tf_experiments.py --speed $v --scale paper50 --no-btoep --data-tag ulatf --comb-spacing 8 --gpu
done
# -> results_subregion_tf_v{0,3,6,10}_ulatf_comb8_paper50/
```
Wall time: space-freq ~66 min; time-freq ~3.0-4.3h per velocity (comb pilot
doesn't change training cost -- EM still fits on the full training channel,
only the CME evaluation step sees the sparser grid).

**Results** (30 dB unless noted). As expected, the DFT/circulant delay
model's aliasing under a subsampled pilot reproduces the paper's Fig. 3
story in both slices: `b-circ`, `2x1D-circ` and the `PDP+DS` baselines hit a
hard NMSE floor (visible flattening past ~10-20 dB in both PNGs above)
instead of continuing to descend with SNR.

*Space-freq (D=256):* `full` is best at **every** SNR from -10 to 30 dB
(2.0e-3 @30dB vs `2x1D` 4.3e-3, `2x1D-toep` 4.8e-3, `kron` 8.6e-3, `2x1D-circ`
2.0e-2, `b-circ` 3.2e-2, `PDP+DS` pair ~7.7-8.5e-2 floor) -- same as the
paper50 MISO comb-8 result earlier in this file.

*Time-freq (D=896):* `full`/`kron` lead at -10..20 dB for every velocity,
but at 30 dB `GMM 2x1D` (and, more weakly, `2x1D-toep`) overtakes `full` at
**every** velocity -- e.g. v=10: `2x1D` 3.5e-3 < `2x1D-toep` 3.9e-3 < `full`
4.0e-3 < `kron` 8.1e-3; v=0: `2x1D` 3.2e-3 < `full` 3.5e-3 < `2x1D-toep`
4.1e-3. This is the same "structured prior beats naive full-pilot-style
inversion under sparse pilots" effect the paper attributes to `b-toep` --
with `b-toep` excluded here (guideline condition 1), the cascade `2x1D`/
`2x1D-toep` pick up that role instead of the jointly-fit `full`/`kron`.
`b-circ`/`2x1D-circ`/`PDP+DS` still floor as usual (e.g. v=10 @30dB: `b-circ`
7.3e-2, `PDP+DS kron` 7.5e-2, `PDP+DS 2x1D` 8.0e-2).

### `GMM Weichselberger` addition (2026-09-29, per `exp2_guideline.md`)

Added a fifth joint structured-covariance estimator, **`GMM Weichselberger`**,
to this velocity sweep's comb-8-family pilot experiments only -- per
`exp2_guideline.md`'s explicit scope: the `results_subregion_tf_v{0,3,6,10}_
ulatf_{circ,unif}_c8t2_paper50/` and `results_subregion_M4_ulatf_comb8_
paper50/` directories (the sparse-pilot-pattern experiments from
`run_subregion_tf_pilot_experiments.py` and the M4 comb-8 space-freq
experiment above), NOT the plain full-pilot / plain-comb8 directories
elsewhere in this file, which were left untouched.

**What it is.** Implemented to the exact spec of
`Weichselberger_GMM_OptionB_Experiment_Guide.pdf` ("Option B: Component-wise
Tx/Rx Eigenbasis Learning"), *not* a hand-rolled simplification -- an earlier
pass (fixed global eigenbasis + diagonal EM, closer to `b-circ` with a
data-derived basis instead of the DFT) was built, smoke-tested, then
discarded once the guide surfaced, and every affected result directory was
restored from backup before the correct version was (re-)run, so nothing
downstream saw the wrong numbers.

The Weichselberger model sits strictly between `full` and `kron` in
structural hierarchy (`Full` &sup; `Weichselberger` &sup; `Kronecker`):
where `kron`'s per-component covariance is a plain Kronecker product
`kron(C_time,k, C_freq,k)` (Tx/Rx mode power forced separable), Weichselberger
allows an arbitrary nonnegative **coupling matrix** `Omega_k` (Nc x Nt)
between a component-specific Rx eigenbasis `Ur,k` (Nc x Nc) and Tx eigenbasis
`Ut,k` (Nt x Nt):

```
C_k = (Ut,k* (x) Ur,k) diag(vec(Omega_k)) (Ut,k* (x) Ur,k)^H
```

"Option B" ("component-wise"): **every component gets its own basis**,
re-derived every EM iteration from that component's *responsibility-weighted*
marginal covariances (not one basis shared by the whole corpus, and not
`b-circ`'s fixed, data-independent DFT basis):

1. **E-step:** transform each sample into the current basis,
   `H~n,k = Ur,k^H Hn Ut,k`, and compute `log p(Hn|k)` directly from
   `Omega_k` (entries of `H~n,k` are modeled as *independent* `CN(0, Omega_k,ij)`
   -- no dense covariance is ever built or inverted here).
2. **M-step A:** responsibility-weighted Rx/Tx marginal covariances
   `R_r,k = (1/N_k) sum_n gamma_nk Hn Hn^H`, `R_t,k = (1/N_k) sum_n gamma_nk Hn^H Hn`.
3. **M-step B:** `Ur,k`/`Ut,k` <- eigenvectors of `R_r,k`/`R_t,k` (descending
   eigenvalues) -- a *moment* update, not an exact joint-likelihood M-step
   (only `Omega_k`'s update, given a basis, is the exact weighted-ML power
   estimate -- the guide is explicit about this; see its "중요한 해석" note).
4. **M-step C:** `Omega_k = (1/N_k) sum_n gamma_nk |Ur,k^H Hn Ut,k|^2`
   (element-wise), floored at a small epsilon.

Zero-mean per component (unlike `full`/`b-toep`/`b-circ`/`kron` here, which
do fit a component mean) -- matches the guide's model (Eq. 1) exactly.
Init follows the guide's no-warm-start path (Sec. 5): k-means++ centers on
the vectorized data, one hard nearest-center assignment, then one moment
step to seed `Ur`/`Ut`/`Omega` before the main E/M loop -- deliberately
avoiding a `full`/`kron` warm start, which would have meant re-fitting an
already-completed estimator just to seed this one.

**Code changes** (`pipeline/GMM_CE/`):
- `gmmce/complex_gmm.py`: `fit_weichselberger_gmm` (CPU reference) plus
  `_weichselberger_moment_step` / `_weichselberger_loglik` /
  `_weichselberger_dense_covs` helpers. Unlike every other `fit_*_gmm` here,
  it takes `H` as `(N, Nc, Nt)` channel matrices, not `vec()`'d -- `Ur,k`/
  `Ut,k` act on the two channel axes directly. Verified against the guide's
  own sanity checks (Sec. 8): power preservation under the unitary
  transform, `Ur,k`/`Ut,k` unitarity, and marginal consistency (`Omega_k`'s
  row/column sums equal `R_r,k`/`R_t,k`'s eigenvalues) all hold to float
  precision on synthetic data.
- `gmmce/gpu_gmm.py`: `fit_weichselberger_gmm` (GPU/torch), same algorithm,
  batched over K wherever the memory allows it (the `R_r,k`/`R_t,k`
  reduction and the eigendecomposition batch over all K components in one
  shot; only the `O(N*Nc*Nt)`-sized per-component transform -- needed for
  both the E-step log-likelihood and the M-step's `Omega_k` -- loops over K,
  since materializing it for every component at once would need a
  `(K,N,Nc,Nt)` tensor, infeasible at paper50 scale). Matches the CPU
  version to ~1e-14 relative error on synthetic data; runs in
  `GMMCE_GPU_STRUCT_DTYPE`'s structured-variant precision (complex64 by
  default, same as `b-circ`/`b-toep`).
- `gmmce/pipeline.py`: `"Weichselberger"` added to `JOINT_VARIANTS`;
  `TrainedModels.gmm_weichselberger`; `train_all` fits it from `H_train`
  directly (not the `vec()`'d `Xtr` every other variant uses).
- `gmmce/plotting.py`: `STYLE["GMM Weichselberger"]` (orange, downward
  triangle).
- `run_subregion_experiments.py` / `run_subregion_tf_experiments.py`:
  `"GMM Weichselberger"` added to `FIG3_ORDER`/`FIG4_ORDER` and to
  `_eval_all_joint_and_cascade` (evaluated exactly like `full`/`kron`/
  `b-circ` via `evaluate_joint` -- the CME/`cme_estimate` code path is
  generic over any dense-covariance GMM, so it needed no changes).
- `run_subregion_tf_pilot_experiments.py`: `"GMM Weichselberger"` added to
  `JOINT_ONLY` (the estimator subset valid under the non-separable
  `circ_c8t2` pilot -- Weichselberger's CME, like `full`/`kron`/`b-circ`,
  only needs an arbitrary selection matrix `A`, not a Cartesian-product
  pilot grid, so it runs under both `unif_c8t2` and `circ_c8t2`).
- **New script `run_weichselberger_addon.py`**: fits *only*
  `GMM Weichselberger` (via `train_all(..., which=("Weichselberger",))`,
  which touches no other variant) at each existing `nmse_vs_snr` /
  `nmse_vs_training` / `nmse_vs_components` x-value, and merges the new
  curve into the *existing* JSON/PNG outputs -- per the guideline's
  explicit condition not to re-run `full`/`kron`/`b-circ`/`2x1D`/etc.
  Verified for every one of the 9 target directories: the merged JSON
  files are byte-identical to the pre-existing ones except for the added
  `"GMM Weichselberger"` key. For the two `tfpilot` patterns per velocity,
  one model is fit per x-value and reused for *both* `unif_c8t2` and
  `circ_c8t2` (the GMM fit never sees the pilot pattern -- same sharing the
  original `run_subregion_tf_pilot_experiments.py` run used).
  ```bash
  cd pipeline/GMM_CE
  python3 run_weichselberger_addon.py --target m4comb8 --gpu
  python3 run_weichselberger_addon.py --target tfpilot --speed 0 3 6 10 --gpu
  ```
  Same `paper50` scale/seed/SNR list/pilot geometry as the pre-existing
  runs (read back from each directory's own JSON, not re-derived).

**Results** (30 dB unless noted; GPU, paper50, same K=128/N_train=1e5/
n_iter=50 settings as every other estimator in this file):

*Space-freq, `results_subregion_M4_ulatf_comb8_paper50/` (D=256, Np=32/256):*
`full` 2.0e-3 < `Weichselberger` 2.5e-3 < `2x1D` 4.3e-3 < `2x1D-toep` 4.8e-3
< `kron` 8.6e-3 < `b-circ` 3.2e-2 -- `Weichselberger` lands a close second to
`full` at every SNR from -10 to 30 dB, clearly ahead of `kron` (its closest
structural relative: same Tx/Rx-eigenbasis idea, but forced-separable
coupling) and `b-circ` (fixed DFT basis instead of a learned one).

*Time-freq, `results_subregion_tf_v{0,3,6,10}_ulatf_{unif,circ}_c8t2_
paper50/` (D=896), `full`/`kron`/`b-circ`/`Weichselberger` @30dB:*

| v (m/s) | pattern | full | kron | b-circ | **Weichselberger** |
|---|---|---|---|---|---|
| 0  | unif (Np=56/896)  | 3.58e-3 | 6.26e-3 | 1.71e-2 | **3.29e-3** |
| 0  | circ (Np=56/896)  | 2.41e-4 | 2.88e-4 | 7.39e-4 | **2.41e-4** |
| 3  | unif | 3.89e-3 | 6.60e-3 | 5.83e-2 | **2.92e-3** |
| 3  | circ | 3.53e-4 | 4.39e-4 | 4.41e-3 | **3.44e-4** |
| 6  | unif | 4.10e-3 | 8.24e-3 | 7.33e-2 | **3.26e-3** |
| 6  | circ | 4.33e-4 | 6.39e-4 | 1.09e-2 | **4.10e-4** |
| 10 | unif | 4.14e-3 | 8.04e-3 | 7.27e-2 | **2.99e-3** |
| 10 | circ | 5.25e-4 | 8.34e-4 | 1.51e-2 | **5.19e-4** |

`Weichselberger` matches or *beats* `full` at every one of the 8
velocity/pattern combinations at 30 dB (the `unif_c8t2` margin is the
largest -- e.g. v=10: 2.99e-3 vs `full`'s 4.14e-3), and clearly beats `kron`
and `b-circ` throughout. This is exactly the pattern the guide's Sec. 7
anticipated: a non-separable, *learned* eigenbasis coupling captures the
ray-traced channel's structure at least as well as an unconstrained `full`
fit, while `kron`'s forced-separable coupling and `b-circ`'s fixed
(non-data-adaptive) DFT basis both leave a visible gap. `b-circ`'s
increasing SNR floor with growing velocity (1.7e-2 @v=0 -> 7.3e-2 @v=10,
`unif_c8t2`) reflects the DFT/circulant delay-model aliasing under a
subsampled pilot documented earlier in this file; `Weichselberger`'s learned
Rx/Tx bases sidestep that entirely.

**Why it wins even at generous training sizes (not just the small-data
regime).** The K-sweep is the cleanest evidence this isn't purely a
sample-size effect: at `K=8` on the `unif_c8t2` `v=3` experiment,
`N_k = N/K = 12500` -- almost 14x `D=896`, comfortably enough to estimate an
unconstrained covariance -- yet `Weichselberger` (1.81e-2) still beats
`full` (2.11e-2) by 14%. The reason: this ray-traced channel's covariance is
extremely low-rank ("specular"). A quick eigendecomposition of the pooled
training covariance shows **99% of the energy in just 8 of 896 dimensions**
(top eigenvalue alone: 46%) for the time-freq slice, vs. 27 of 256 for the
space-freq slice -- consistent with this file's earlier documented
`~25/256` figure for the plain M4 experiment, and with why `full` eventually
*does* catch up there (once `N_k` exceeds `D=256`) but not in the D=896
time-freq case tested here. An unconstrained `full` covariance still has to
estimate all ~888 near-null directions; sampling noise leaks small spurious
values into them, and the CME's `(A C A^H + C_n)^{-1}` amplifies that noise
most at high SNR (small `C_n`). `Weichselberger`'s covariance, built from
only a 64-dim + 14-dim eigenbasis pair plus a coupling matrix, has no room
for that -- the same "near-zero bias, large variance reduction" story this
file already documents for `b-toep`, just via a *learned* basis instead of
an assumed Toeplitz/WSS one.

### Full-pilot extension (2026-09-29)

Same estimator, same guideline condition (don't re-run `full`/`kron`/
`b-circ`/`2x1D`/etc.), extended to the **full-pilot** (`A = I`) paper50
directories from the main velocity sweep above:
`results_subregion_M4_ulatf_paper50/` and
`results_subregion_tf_v{0,3,6,10}_ulatf_paper50/` (5 directories; the
`comb8`/`c8t2` sparse-pilot directories above are untouched by this pass).
```bash
cd pipeline/GMM_CE
python3 run_weichselberger_addon.py --target m4fullpilot --gpu
python3 run_weichselberger_addon.py --target tffullpilot --speed 0 3 6 10 --gpu
```
Verified the same way: every merged JSON is byte-identical to the
pre-existing file except for the added `"GMM Weichselberger"` key.

**A real precision bug surfaced and was fixed.** The initial full-pilot
`nmse_vs_snr` numbers showed `Weichselberger` losing to `full` -- and even
to `kron` -- specifically and only at 30 dB (ratio to `full`: ~0.95-1.01 at
-10..20 dB, jumping to 1.7-2.3x at 30 dB, for every one of the 4 time-freq
velocities -- a sharp jump at the single highest-SNR point, not a smooth
trend, and the tell-tale shape of a precision floor rather than a modeling
one). This is the **same complex64/fp32 CME precision issue this file
already documents for `GMM b-toep`** at this D=896, K=128 scale (full pilot
means the CME's `A C A^H + C_n` matrix is the full `D x D` size, and at
30 dB `C_n` is small enough that fp32 can't resolve it cleanly) --
`gmmce/gpu_gmm.py` runs every structured variant (`b-toep`, `b-circ`, and
now `Weichselberger`) in complex64 by default for speed. Confirmed by
re-fitting with `GMMCE_GPU_STRUCT_DTYPE=complex128`: the 30 dB point moved
from 6.09e-5 to 3.47e-5 for v=10 (matching `full`'s 3.55e-5 almost exactly),
with the -10..20 dB points and every other estimator's numbers unchanged
(confirmed by diff). The sparse-pilot (`comb8`/`c8t2`) results above were
**not** affected -- there the CME's matrix is only `Np x Np` (56 or 32, not
896/256), small enough that fp32 stays well-conditioned even at 30 dB.
**Fix applied**: only the `nmse_vs_snr` curve was re-fit at complex128 (the
`nmse_vs_training`/`nmse_vs_components` sweeps use `SNR=10dB`, well below
where the floor appears, and were confirmed unaffected) for all 5
directories; `nmse_vs_training`/`nmse_vs_components` were left untouched.

**Results** (30 dB, complex128-corrected; K=128/N_train=1e5/n_iter=50 as
everywhere else):

| experiment | full | kron | Weichselberger |
|---|---|---|---|
| M4 space-freq (D=256) | 1.38e-4 | 3.10e-4 | 1.83e-4 |
| tf v=0 (D=896) | 2.11e-5 | 2.37e-5 | **2.13e-5** |
| tf v=3 | 2.84e-5 | 3.27e-5 | **2.77e-5** |
| tf v=6 | 3.22e-5 | 3.95e-5 | **3.12e-5** |
| tf v=10 | 3.55e-5 | 4.37e-5 | **3.47e-5** |

Every time-freq velocity: `Weichselberger` ties or slightly beats `full`
at 30 dB under full pilot too (not just the sparse-pilot case), and clearly
beats `kron` throughout. `nmse_vs_training`/`nmse_vs_components` show the
same pattern as the sparse-pilot case -- e.g. tf v=10, `N_train=100000`:
`full` 1.45e-3 vs `Weichselberger` 1.38e-3; `K=8`: `full` 2.11e-3 vs
`Weichselberger` 1.91e-3 -- `Weichselberger` wins at every point on both
sweeps, for the same low-effective-rank reason discussed above. Only the
M4 space-freq case shows a persistent (non-precision) gap to `full`
(1.83e-4 vs 1.38e-4 at 30 dB) -- consistent with that slice's milder rank
concentration (27/256 vs 8/896) making the structural constraint cost a
little real bias there, exactly as the training-size crossover for that
same experiment already showed above.

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
