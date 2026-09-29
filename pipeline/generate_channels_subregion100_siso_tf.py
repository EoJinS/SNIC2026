"""
Ray-trace SISO TIME-FREQUENCY channels for ``subregion100`` (x in [-80,20],
y in [-60,40], 0.25 m grid): a SINGLE TX antenna, F subcarriers x S OFDM
symbols per position -- the axes Fesl et al. actually use
(Nc <- subcarrier, Nt <- OFDM symbol), instead of the (subcarrier,
TX-antenna) proxy ``generate_channels_subregion100_marray.py`` builds.

One ``PathSolver()`` call per position gives all S OFDM-symbol snapshots at
once via Sionna's analytic Doppler time-evolution: attach a velocity to the
``Receiver`` and read off ``Paths.cfr(frequencies, sampling_frequency=
1/symbol_dt, num_time_steps=S)`` -- exactly the mechanism
``legacy/pipeline/generate_channels_mobility.py`` already validated for a
per-SUBFRAME antenna-stacked axis; here it drives a per-OFDM-SYMBOL SISO
axis instead. UE speed is fixed (default 10 m/s, matching this project's
own previously-vetted "meaningful, non-degenerate" mobility speed -- see
README) with a random heading per position (no separate velocity axis is
requested here, unlike Fesl's own v-sweep).

Same POSTECH scene physics as ``generate_channels_subregion100.py`` (fc
3.5 GHz, TX (-70,-25,15), diffuse on, max_depth 5, samples_per_src 1e6,
seed 42) -- only the RX gains a velocity and the CFR gains a time axis.
Frequency axis stays at F=64 subcarriers / SCS=240 kHz (``config.py``,
same as the rest of this pipeline). The OFDM SYMBOL duration used for the
slow/time axis is Fesl's own paper value (71.4 us, a 15 kHz-numerology NR
slot symbol incl. CP -- see ``gmmce/channel_model.py``'s
``SystemConfig.symbol_duration_s``), so an S=14-symbol slot spans about
1 ms. This is legitimate and decoupled from the frequency axis's SCS by
design: ``Paths.cfr``'s ``sampling_frequency`` (for ``num_time_steps``) is
an independent parameter of the analytic-Doppler CFR formula
``h(f,t) = sum_i a_i * exp(-j2*pi*f*tau_i) * exp(j2*pi*f_Delta,i*t)`` --
see ``legacy/pipeline/doppler_cube.py``. At v=10 m/s / fc=3.5 GHz this
gives ~40 deg of Doppler phase rotation across the whole S=14-symbol slot
(max f_D=116.7 Hz), i.e. clearly time-selective without being degenerate
-- the same order of magnitude this project already validated for v=10 m/s
over a ~1 ms window (``legacy/pipeline/generate_channels_mobility.py``'s
per-subframe steps).

vec order: ``h[p].reshape(S, F)``, index = ``t*F + f`` (OFDM-symbol-major,
subcarrier-minor) -- exactly ``gmmce/pilots.py``'s flat ``vec(H)`` index
``h[t*Nc + c]`` with ``Nt <- S`` (time), ``Nc <- F`` (freq); the
``(N, Nc, Nt)`` adapter (``gmmce/subregion_tf_data.py``) still applies the
same reshape+transpose ``subregion_data.py`` does, just with
``(symbol, freq)`` as the raw storage axes instead of ``(antenna, freq)``.

Writes BOTH pools the SISO time-freq GMM_CE experiment needs:
  train_siso_S{s}_F{f}_v{v}_subregion100.npz
  test_siso_S{s}_F{f}_v{v}_subregion100.npz  (disjoint grid positions)

    CUDA_VISIBLE_DEVICES=2 python3 -u generate_channels_subregion100_siso_tf.py --n-train 110000 --n-test 12000
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


def build_scene(positions, velocities):
    scene = load_scene(POSTECH_SCENE)
    for name, val in SCATTERING_COEFFICIENTS.items():
        if name in scene.radio_materials:
            scene.radio_materials[name].scattering_coefficient = val
    scene.frequency = FC
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.add(Transmitter(name="tx", position=TX_POS))
    for i, (p, v) in enumerate(zip(positions, velocities)):
        scene.add(Receiver(name=f"rx_{i}",
                           position=[float(p[0]), float(p[1]), float(p[2])],
                           velocity=[float(v[0]), float(v[1]), 0.0]))
    return scene


def ray_trace_chunk(freqs, slow_fs, S, positions, velocities):
    """Returns h: (n_rx, S*F) complex64, vec order h[t*F+f]
    (OFDM-symbol-major, subcarrier-minor) -- gmmce/pilots.py's convention
    with Nt<-S, Nc<-F."""
    scene = build_scene(positions, velocities)
    paths = PathSolver()(scene=scene, max_depth=MAX_DEPTH, samples_per_src=SAMPLES_PER_SRC,
                         los=True, specular_reflection=True, diffuse_reflection=True,
                         refraction=True, synthetic_array=True, seed=SEED_RT)
    # cfr: [num_rx, num_rx_ant(1), num_tx(1), num_tx_ant(1), num_time(S), F]
    cfr = np.asarray(paths.cfr(frequencies=freqs, sampling_frequency=slow_fs,
                               num_time_steps=S, normalize_delays=False,
                               normalize=False, out_type="numpy"))
    h = cfr[:, 0, 0, 0, :, :]                       # (n_rx, S, F)
    n_rx = h.shape[0]
    return np.ascontiguousarray(h).reshape(n_rx, S * freqs.shape[0]).astype(np.complex64)


def _velocities(n, speed, heading_deg, seed):
    rng = np.random.default_rng(seed)
    if heading_deg is not None:
        theta = np.full(n, np.deg2rad(heading_deg))
    else:
        theta = rng.uniform(0.0, 2 * np.pi, size=n)
    vel = speed * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return vel, theta


def _raytrace(positions, velocities, freqs, slow_fs, S, chunk, tag):
    n = len(positions)
    h = np.empty((n, S * len(freqs)), np.complex64)
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        h[lo:hi] = ray_trace_chunk(freqs, slow_fs, S, positions[lo:hi], velocities[lo:hi])
        el = time.time() - t0
        rate = hi / max(el, 1e-9)
        print(f"  [{tag}] {hi}/{n}  ({el:.0f}s, {rate:.1f} pos/s, "
              f"ETA {(n - hi) / max(rate, 1e-9) / 60:.1f} min)", flush=True)
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f", type=int, default=NUM_SUBCARRIERS, help="Nc: subcarriers")
    ap.add_argument("--scs", type=float, default=SUBCARRIER_SPACING, help="OFDM SCS, Hz (freq axis)")
    ap.add_argument("--n-symbols", type=int, default=14, help="Nt: OFDM symbols per slot")
    ap.add_argument("--symbol-dt", type=float, default=SYMBOL_DT_DEFAULT,
                    help="seconds/OFDM symbol (time axis; decoupled from --scs, see module docstring)")
    ap.add_argument("--speed", type=float, default=10.0, help="UE speed, m/s (0 = static baseline)")
    ap.add_argument("--heading", type=float, default=None, help="deg; default = random per position")
    ap.add_argument("--n-train", type=int, default=110000)
    ap.add_argument("--n-test", type=int, default=12000)
    ap.add_argument("--chunk-size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: cap n-train/n-test")
    args = ap.parse_args()

    freqs = np.asarray(subcarrier_frequencies(args.f, args.scs))
    slow_fs = 1.0 / args.symbol_dt

    # drjit caps intermediate tensors at 2**32 entries; cfr builds
    # ~ (chunk * S * F * num_paths). POSTECH + diffuse ~= 1k-1.3k paths;
    # leave headroom (matches legacy/pipeline/generate_channels_mobility.py).
    est_paths = 1400
    max_chunk = int(2.4e9 / max(1, args.n_symbols * args.f * est_paths))
    chunk = args.chunk_size
    if chunk > max_chunk:
        chunk = max(40, max_chunk)
        print(f"[siso-tf] chunk {args.chunk_size} -> {chunk}  (drjit 2^32 tensor limit: "
              f"chunk*S*F*paths < 2^32, S={args.n_symbols} F={args.f})")

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
    print(f"[siso-tf] Nc(freq)={args.f} SCS={args.scs / 1e3:g}kHz  Nt(sym)={args.n_symbols} "
          f"symbol_dt={args.symbol_dt * 1e6:.2f}us  slot={slot_s * 1e3:.3f}ms  D={args.f * args.n_symbols}")
    print(f"[siso-tf] speed={args.speed} m/s  heading="
          f"{'fixed %.0f deg' % args.heading if args.heading is not None else 'random/position'}  "
          f"max f_D={fD:.1f}Hz  slot phase={np.degrees(2 * np.pi * fD * slot_s):.1f} deg")
    print(f"train: {len(train_pos)} random grid positions  |  test: {len(test_pos)} disjoint grid positions")

    for tag, positions, seed_off in (("train", train_pos, 0), ("test", test_pos, 1)):
        vel, theta = _velocities(len(positions), args.speed, args.heading, args.seed + 1000 + seed_off)
        print(f"ray tracing {tag.upper()} ...", flush=True)
        h = _raytrace(positions, vel, freqs, slow_fs, args.n_symbols, chunk, tag)
        out = (f"{OUT_DIR}/{tag}_siso_S{args.n_symbols}_F{args.f}_v{int(args.speed)}_subregion100.npz")
        np.savez(out, h=h, positions=positions, velocities=vel, headings=theta,
                 speed=args.speed, n_subcarriers=args.f, n_symbols=args.n_symbols,
                 subcarrier_spacing=args.scs, symbol_dt=args.symbol_dt, carrier_freq=FC,
                 bbox=bbox, vec_order="h[t*F+f]  (OFDM-symbol-major, subcarrier-minor)")
        print(f"saved -> {out}  ({h.shape[0]} x {h.shape[1]}, mean|h|^2={np.mean(np.abs(h) ** 2):.3e})",
              flush=True)

    os._exit(0)   # drjit/mitsuba teardown segfaults harmlessly after save


if __name__ == "__main__":
    main()
