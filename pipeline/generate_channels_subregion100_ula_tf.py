"""
Ray-trace a single 1xM TX ULA MOBILITY channel cube for ``subregion100``
(x in [-80,20], y in [-60,40], 0.25 m grid): M TX antennas x F subcarriers
x S OFDM symbols per position -- the ONE system that both
``run_subregion_experiments.py`` (space-frequency: antenna x freq) and
``run_subregion_tf_experiments.py`` (time-frequency: freq x OFDM symbol)
are meant to be two 2-D slices of (see guideline.md's 2026-09-16 note):
a real ``M``-element ULA (unlike ``generate_channels_subregion100_siso_tf.py``,
which only ever ray-traces a single standalone antenna -- not an element of
an actual array) that ALSO carries the OFDM-symbol / Doppler time axis
(unlike ``generate_channels_subregion100_marray.py``, which has no time
axis at all). The GMM-CE estimator library only ever sees a generic 2-D
(Nc, Nt) pair (``gmmce/pipeline.py``), so a true 3-D (antenna, freq, time)
estimator isn't supported -- ``derive_ula_tf_slices.py`` slices this cube
down to the two 2-D pools each experiment driver actually consumes:
  - space-freq (antenna x freq): the OFDM-symbol-0 slice -- Doppler phase
    is 0 by construction at t=0, so this slice is IDENTICAL for every
    velocity; still ray-traced once at ``--speed 0`` for a clean,
    unambiguous provenance (no reliance on that invariance holding
    exactly under Sionna's analytic-Doppler CFR normalization).
  - time-freq (freq x OFDM symbol): one fixed TX antenna's slice, per
    velocity (a single antenna element of the SAME 4-element array --
    physically distinct from a standalone SISO antenna at the same
    position, since sub-wavelength element offset changes the multipath
    phase).

TX array: identical to ``generate_channels_subregion100_marray.py`` --
PlanarArray(1, M, spacing 0.5, iso, V-pol). Time/Doppler mechanism:
identical to ``generate_channels_subregion100_siso_tf.py`` -- one
``PathSolver()`` call per position gives all S OFDM-symbol snapshots via
Sionna's analytic Doppler (``Receiver(velocity=...)`` +
``Paths.cfr(sampling_frequency=1/symbol_dt, num_time_steps=S)``).
Same POSTECH scene physics as every other script here (fc 3.5 GHz, TX
(-70,-25,15), diffuse on, max_depth 5, samples_per_src 1e6, seed 42).

Positions: the SAME train/test position pools (same permutation, seeded
independently of ``--speed``) are used across every ``--speed`` value, so
velocity is the only thing that changes between runs of this script.

vec order: ``h[p].reshape(M, S, F)``, index = ``m*S*F + t*F + f``
(antenna-major, then OFDM-symbol, then subcarrier-minor).

Writes:
  train_ula{M}_S{S}_F{F}_v{speed}_subregion100.npz
  test_ula{M}_S{S}_F{F}_v{speed}_subregion100.npz

    CUDA_VISIBLE_DEVICES=2 python3 -u generate_channels_subregion100_ula_tf.py --speed 0   # space-freq source (once)
    CUDA_VISIBLE_DEVICES=2 python3 -u generate_channels_subregion100_ula_tf.py --speed 3
    CUDA_VISIBLE_DEVICES=2 python3 -u generate_channels_subregion100_ula_tf.py --speed 6
    CUDA_VISIBLE_DEVICES=2 python3 -u generate_channels_subregion100_ula_tf.py --speed 10
Then ``derive_ula_tf_slices.py`` builds the two per-experiment 2-D pools.
"""
import argparse
import os
import time

import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
from sionna.rt.utils import subcarrier_frequencies

from scene_geometry import POSTECH_SCENE                     # pins CUDA_VISIBLE_DEVICES=2
from config import NUM_SUBCARRIERS, SUBCARRIER_SPACING

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

C_LIGHT = 299_792_458.0
TX_POS = [-70.0, -25.0, 15.0]
FC = 3.5e9
MAX_DEPTH = 5
SAMPLES_PER_SRC = 1_000_000
SEED_RT = 42
SCATTERING_COEFFICIENTS = {"itu_concrete": 0.1, "itu_brick": 0.15, "itu_very_dry_ground": 0.05}
SYMBOL_DT_DEFAULT = 71.4e-6   # Fesl et al.'s own OFDM symbol duration (15 kHz SCS + CP)

POS_NPZ = "../Data/positions_subregion_grid_0.25m_100x100.npz"
OUT_DIR = "../Data"


def build_scene(m, positions, velocities):
    scene = load_scene(POSTECH_SCENE)
    for name, val in SCATTERING_COEFFICIENTS.items():
        if name in scene.radio_materials:
            scene.radio_materials[name].scattering_coefficient = val
    scene.frequency = FC
    scene.tx_array = PlanarArray(num_rows=1, num_cols=m, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.add(Transmitter(name="tx", position=TX_POS))
    for i, (p, v) in enumerate(zip(positions, velocities)):
        scene.add(Receiver(name=f"rx_{i}",
                           position=[float(p[0]), float(p[1]), float(p[2])],
                           velocity=[float(v[0]), float(v[1]), 0.0]))
    return scene


def ray_trace_chunk(m, freqs, slow_fs, S, positions, velocities):
    """Returns h: (n_rx, M*S*F) complex64, vec order h[a*S*F + t*F + f]
    (antenna-major, OFDM-symbol-mid, subcarrier-minor)."""
    scene = build_scene(m, positions, velocities)
    paths = PathSolver()(scene=scene, max_depth=MAX_DEPTH, samples_per_src=SAMPLES_PER_SRC,
                         los=True, specular_reflection=True, diffuse_reflection=True,
                         refraction=True, synthetic_array=True, seed=SEED_RT)
    # cfr: [num_rx, num_rx_ant(1), num_tx(1), num_tx_ant(M), num_time(S), F]
    cfr = np.asarray(paths.cfr(frequencies=freqs, sampling_frequency=slow_fs,
                               num_time_steps=S, normalize_delays=False,
                               normalize=False, out_type="numpy"))
    h = cfr[:, 0, 0, :, :, :]                       # (n_rx, M, S, F)
    n_rx = h.shape[0]
    return np.ascontiguousarray(h).reshape(n_rx, m * S * freqs.shape[0]).astype(np.complex64)


def _velocities(n, speed, heading_deg, seed):
    rng = np.random.default_rng(seed)
    if speed == 0:
        return np.zeros((n, 2)), np.zeros(n)
    if heading_deg is not None:
        theta = np.full(n, np.deg2rad(heading_deg))
    else:
        theta = rng.uniform(0.0, 2 * np.pi, size=n)
    vel = speed * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return vel, theta


def _raytrace(m, positions, velocities, freqs, slow_fs, S, chunk, tag):
    n = len(positions)
    h = np.empty((n, m * S * len(freqs)), np.complex64)
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        h[lo:hi] = ray_trace_chunk(m, freqs, slow_fs, S, positions[lo:hi], velocities[lo:hi])
        el = time.time() - t0
        rate = hi / max(el, 1e-9)
        print(f"  [{tag}] {hi}/{n}  ({el:.0f}s, {rate:.1f} pos/s, "
              f"ETA {(n - hi) / max(rate, 1e-9) / 60:.1f} min)", flush=True)
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=4, help="TX ULA columns (1xM array)")
    ap.add_argument("--f", type=int, default=NUM_SUBCARRIERS, help="Nc: subcarriers")
    ap.add_argument("--scs", type=float, default=SUBCARRIER_SPACING, help="OFDM SCS, Hz (freq axis)")
    ap.add_argument("--n-symbols", type=int, default=14, help="Nt: OFDM symbols per slot")
    ap.add_argument("--symbol-dt", type=float, default=SYMBOL_DT_DEFAULT,
                    help="seconds/OFDM symbol (time axis; decoupled from --scs)")
    ap.add_argument("--speed", type=float, default=10.0, help="UE speed, m/s (0 = static, no Doppler)")
    ap.add_argument("--heading", type=float, default=None, help="deg; default = random per position")
    ap.add_argument("--n-train", type=int, default=110000)
    ap.add_argument("--n-test", type=int, default=12000)
    ap.add_argument("--chunk-size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42,
                    help="position permutation seed -- kept fixed across --speed values so "
                         "every velocity ray-traces the SAME train/test positions")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: cap n-train/n-test")
    args = ap.parse_args()

    freqs = np.asarray(subcarrier_frequencies(args.f, args.scs))
    slow_fs = 1.0 / args.symbol_dt

    est_paths = 1400
    max_chunk = int(2.4e9 / max(1, args.m * args.n_symbols * args.f * est_paths))
    chunk = args.chunk_size
    if chunk > max_chunk:
        chunk = max(20, max_chunk)
        print(f"[ula-tf] chunk {args.chunk_size} -> {chunk}  (drjit 2^32 tensor limit: "
              f"chunk*M*S*F*paths < 2^32, M={args.m} S={args.n_symbols} F={args.f})")

    rng = np.random.default_rng(args.seed)
    pos_data = np.load(POS_NPZ)
    grid_pos = np.asarray(pos_data["positions"], np.float64)
    bbox = np.asarray(pos_data["bbox"], float)
    perm = rng.permutation(len(grid_pos))
    n_train = args.limit or args.n_train
    n_test = args.limit or args.n_test
    if n_train + n_test > len(grid_pos):
        raise ValueError(f"grid has {len(grid_pos)} positions; need {n_train}+{n_test} disjoint")
    train_pos = grid_pos[np.sort(perm[:n_train])]
    test_pos = grid_pos[np.sort(perm[n_train:n_train + n_test])]

    lam = C_LIGHT / FC
    fD = args.speed / lam
    slot_s = args.n_symbols * args.symbol_dt
    print(f"[ula-tf] M(ant)={args.m} Nc(freq)={args.f} SCS={args.scs / 1e3:g}kHz  Nt(sym)={args.n_symbols} "
          f"symbol_dt={args.symbol_dt * 1e6:.2f}us  slot={slot_s * 1e3:.3f}ms  D_cube={args.m * args.f * args.n_symbols}")
    print(f"[ula-tf] speed={args.speed} m/s  heading="
          f"{'fixed %.0f deg' % args.heading if args.heading is not None else ('n/a (static)' if args.speed == 0 else 'random/position')}  "
          f"max f_D={fD:.1f}Hz  slot phase={np.degrees(2 * np.pi * fD * slot_s):.1f} deg")
    print(f"train: {len(train_pos)} random grid positions  |  test: {len(test_pos)} disjoint grid positions")

    for tag, positions, seed_off in (("train", train_pos, 0), ("test", test_pos, 1)):
        vel, theta = _velocities(len(positions), args.speed, args.heading, args.seed + 1000 + seed_off)
        print(f"ray tracing {tag.upper()} ...", flush=True)
        h = _raytrace(args.m, positions, vel, freqs, slow_fs, args.n_symbols, chunk, tag)
        out = (f"{OUT_DIR}/{tag}_ula{args.m}_S{args.n_symbols}_F{args.f}_v{int(args.speed)}_subregion100.npz")
        np.savez(out, h=h, positions=positions, velocities=vel, headings=theta,
                 speed=args.speed, m_tx=args.m, n_subcarriers=args.f, n_symbols=args.n_symbols,
                 subcarrier_spacing=args.scs, symbol_dt=args.symbol_dt, carrier_freq=FC,
                 bbox=bbox, vec_order="h[a*S*F+t*F+f]  (antenna-major, OFDM-symbol-mid, subcarrier-minor)")
        print(f"saved -> {out}  ({h.shape[0]} x {h.shape[1]}, mean|h|^2={np.mean(np.abs(h) ** 2):.3e})",
              flush=True)

    os._exit(0)   # drjit/mitsuba teardown segfaults harmlessly after save


if __name__ == "__main__":
    main()
