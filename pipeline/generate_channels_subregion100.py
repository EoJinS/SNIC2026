"""
Ray-trace M=16 / F=64 channels for the DENSE 0.25 m sub-region grid
(``positions_subregion_grid_0.25m_100x100.npz``, bbox x=[-80,20]
y=[-60,40], 123,583 outdoor positions) and write a training dataset in
the exact same layout / physics as ``train_M16_F64_subregion.npz`` so the
existing Share pipeline (``exp_nmse_vs_training_size.py --data-tag
subregion100``) can EM-fit GMMs on the restricted, denser map.

Physics matches the rest of this pipeline (see ``exp_trajectory_nmse.py``
and ``mobility_leg/generate_grid_channels.py``):
  - fc 3.5 GHz, SCS 240 kHz, 64 active subcarriers (fft_size 64, no guard)
  - TX PlanarArray 1x16, iso / V, 0.5 spacing; RX 1x1 iso / V
  - TX at (-70, -25, 15); diffuse reflection ON with per-material
    scattering coefficients; PathSolver max_depth=5, refraction on
  - cfr(normalize_delays=False, normalize=False) -- real physical scale
  - h stored ANTENNA-major / subcarrier-minor: h[p].reshape(M, F), i.e.
    index = m*F + f  -- identical to generate_dataset_subregion.py's
    ``(a @ freq_phase).reshape(-1)`` convention.

Chunked + checkpointed: each chunk of receivers is ray traced and the
partial ``h`` is flushed to ``train_M16_F64_subregion100.partial.npz``
so an interrupted run resumes (``--resume``) instead of restarting.

Run from ``Share/pipeline/`` in the sionna-rt env:
    CUDA_VISIBLE_DEVICES=2 python3 -u generate_channels_subregion100.py
"""
import argparse
import os
import time

import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
from sionna.rt.utils import subcarrier_frequencies

from scene_geometry import POSTECH_SCENE
from config import DEFAULT_M_TX, NUM_SUBCARRIERS, SUBCARRIER_SPACING

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

TX_POS = [-70.0, -25.0, 15.0]
FC = 3.5e9
MAX_DEPTH = 5
SAMPLES_PER_SRC = 1_000_000
SEED = 42
SCATTERING_COEFFICIENTS = {"itu_concrete": 0.1, "itu_brick": 0.15, "itu_very_dry_ground": 0.05}

POS_NPZ = "../Data/positions_subregion_grid_0.25m_100x100.npz"
OUT_NPZ = "../Data/train_M16_F64_subregion100.npz"
PARTIAL_NPZ = "../Data/train_M16_F64_subregion100.partial.npz"


def build_scene(m, f, positions):
    scene = load_scene(POSTECH_SCENE)
    for name, value in SCATTERING_COEFFICIENTS.items():
        if name in scene.radio_materials:
            scene.radio_materials[name].scattering_coefficient = value
    scene.frequency = FC
    scene.bandwidth = f * SUBCARRIER_SPACING
    scene.tx_array = PlanarArray(num_rows=1, num_cols=m, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.add(Transmitter(name="tx", position=TX_POS))
    for i, p in enumerate(positions):
        scene.add(Receiver(name=f"rx_{i}", position=[float(p[0]), float(p[1]), float(p[2])]))
    return scene


def ray_trace_chunk(m, f, freqs, positions):
    scene = build_scene(m, f, positions)
    solver = PathSolver()
    paths = solver(scene=scene, 
                   max_depth=MAX_DEPTH, 
                   samples_per_src=SAMPLES_PER_SRC,
                   los=True, 
                   specular_reflection=True, 
                   diffuse_reflection=True,
                   refraction=True, 
                   synthetic_array=True, 
                   seed=SEED)
    # cfr: [num_rx, num_rx_ant(1), num_tx(1), num_tx_ant(M), num_time(1), F]
    cfr = paths.cfr(frequencies=freqs, 
                    sampling_frequency=SUBCARRIER_SPACING,
                    num_time_steps=1, 
                    normalize_delays=False, 
                    normalize=False,
                    out_type="numpy")
    h = np.asarray(cfr)[:, 0, 0, :, 0, :]          # [num_rx, M, F]
    return h.reshape(h.shape[0], m * f).astype(np.complex64)   # antenna-major / subcarrier-minor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=DEFAULT_M_TX)
    parser.add_argument("--f", type=int, default=NUM_SUBCARRIERS)
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--resume", action="store_true",
                        help="continue from train_M16_F64_subregion100.partial.npz")
    parser.add_argument("--limit", type=int, default=None,
                        help="only ray trace the first N positions (smoke test)")
    args = parser.parse_args()

    freqs = np.asarray(subcarrier_frequencies(args.f, SUBCARRIER_SPACING))
    pos_data = np.load(POS_NPZ)
    positions = np.asarray(pos_data["positions"], dtype=np.float64)
    bbox = np.asarray(pos_data["bbox"], dtype=float)   # x_min, x_max, y_min, y_max
    if args.limit is not None:
        positions = positions[: args.limit]
    n = len(positions)

    start = 0
    h_all = np.empty((n, args.m * args.f), dtype=np.complex64)
    if args.resume and os.path.exists(PARTIAL_NPZ):
        prev = np.load(PARTIAL_NPZ)
        done = int(prev["n_done"])
        h_all[:done] = prev["h"][:done]
        start = done
        print(f"resuming: {done}/{n} positions already ray traced")

    print(f"ray tracing {n} positions (M={args.m}, F={args.f}) in chunks of "
          f"{args.chunk_size}, max_depth={MAX_DEPTH}, samples_per_src={SAMPLES_PER_SRC} ...")
    t0 = time.time()
    for lo in range(start, n, args.chunk_size):
        hi = min(lo + args.chunk_size, n)
        h_all[lo:hi] = ray_trace_chunk(args.m, args.f, freqs, positions[lo:hi])
        el = time.time() - t0
        rate = (hi - start) / max(el, 1e-9)
        eta = (n - hi) / max(rate, 1e-9)
        print(f"  {hi}/{n} done  ({el:.0f}s elapsed, {rate:.1f} pos/s, ETA {eta/60:.1f} min)",
              flush=True)
        np.savez(PARTIAL_NPZ, h=h_all, n_done=hi, positions=positions)

    np.savez(OUT_NPZ, h=h_all, positions=positions, bbox=bbox,
             m_tx=args.m, n_subcarriers=args.f, subcarrier_spacing=SUBCARRIER_SPACING,
             carrier_freq=FC, n_per_position=1, sampling="single_real_per_position")
    print(f"saved -> {OUT_NPZ}  ({h_all.shape[0]} samples, N={h_all.shape[1]})")
    if os.path.exists(PARTIAL_NPZ):
        os.remove(PARTIAL_NPZ)


if __name__ == "__main__":
    main()
