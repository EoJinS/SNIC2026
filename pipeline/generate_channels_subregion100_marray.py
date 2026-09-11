"""
Ray-trace ``subregion100`` (x in [-80,20], y in [-60,40], 0.25 m grid)
channels for an ARBITRARY TX ULA size ``--m`` (e.g. 1x4), in the exact
same physics / vec layout as ``generate_channels_subregion100.py``
(fc 3.5 GHz, SCS 240 kHz, F=64, TX (-70,-25,15), max_depth 5,
samples_per_src 1e6, diffuse on, seed 42; h stored antenna-major
h[p].reshape(M, F), index = m*F + f).

Writes BOTH pools the GMM_CE experiment needs:
  train_M{m}_F64_subregion100.npz  -- random subsample of the dense grid
  test_M{m}_F64_subregion100.npz   -- the SAME held-out positions as
                                      test_M16_F64_subregion100.npz
                                      (so results stay comparable)

    CUDA_VISIBLE_DEVICES=2 python3 -u generate_channels_subregion100_marray.py --m 4 --n-train 20000
"""
import argparse
import os

import numpy as np

from scene_geometry import POSTECH_SCENE                     # pins CUDA_VISIBLE_DEVICES=2
from config import NUM_SUBCARRIERS, SUBCARRIER_SPACING
from generate_channels_subregion100 import ray_trace_chunk, FC, MAX_DEPTH, SAMPLES_PER_SRC
from sionna.rt.utils import subcarrier_frequencies

POS_NPZ = "../Data/positions_subregion_grid_0.25m_100x100.npz"
TEST_M16 = "../Data/test_M16_F64_subregion100.npz"
OUT_DIR = "../Data"


def _raytrace(positions, m, f, freqs, chunk):
    h = np.empty((len(positions), m * f), np.complex64)
    for lo in range(0, len(positions), chunk):
        hi = min(lo + chunk, len(positions))
        h[lo:hi] = ray_trace_chunk(m, f, freqs, positions[lo:hi])
        print(f"  {hi}/{len(positions)}", flush=True)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True, help="TX ULA columns (1xM array)")
    ap.add_argument("--f", type=int, default=NUM_SUBCARRIERS)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=0,
                    help="0 -> test pool = the M16 held-out positions (2938); "
                         ">0 -> draw N disjoint grid positions as an independent test pool")
    ap.add_argument("--chunk-size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    freqs = np.asarray(subcarrier_frequencies(args.f, SUBCARRIER_SPACING))
    rng = np.random.default_rng(args.seed)

    pos_data = np.load(POS_NPZ)
    grid_pos = np.asarray(pos_data["positions"], np.float64)
    bbox = np.asarray(pos_data["bbox"], float)
    perm = rng.permutation(len(grid_pos))
    idx = perm[: args.n_train]
    train_pos = grid_pos[np.sort(idx)]

    if args.n_test > 0:
        test_idx = perm[args.n_train: args.n_train + args.n_test]     # disjoint from train
        if len(test_idx) < args.n_test:
            raise ValueError(f"grid has {len(grid_pos)}; need {args.n_train}+{args.n_test}")
        test_pos = grid_pos[np.sort(test_idx)]
        test_desc = f"{len(test_pos)} disjoint grid positions"
    else:
        test_pos = np.asarray(np.load(TEST_M16)["positions"], np.float64)
        test_desc = f"{len(test_pos)} (== M16 held-out)"

    print(f"[M={args.m} F={args.f}] physics: fc={FC/1e9}GHz max_depth={MAX_DEPTH} "
          f"samples_per_src={SAMPLES_PER_SRC:.0e}")
    print(f"train: {len(train_pos)} random grid positions  |  test: {test_desc}")

    print("ray tracing TRAIN ...", flush=True)
    h_tr = _raytrace(train_pos, args.m, args.f, freqs, args.chunk_size)
    tr = f"{OUT_DIR}/train_M{args.m}_F{args.f}_subregion100.npz"
    np.savez(tr, h=h_tr, positions=train_pos, bbox=bbox, m_tx=args.m, n_subcarriers=args.f,
             subcarrier_spacing=SUBCARRIER_SPACING, carrier_freq=FC, n_per_position=1,
             sampling="single_real_per_position")
    print(f"saved -> {tr}  ({h_tr.shape[0]} x {h_tr.shape[1]})", flush=True)

    print("ray tracing TEST ...", flush=True)
    h_te = _raytrace(test_pos, args.m, args.f, freqs, args.chunk_size)
    te = f"{OUT_DIR}/test_M{args.m}_F{args.f}_subregion100.npz"
    np.savez(te, h=h_te, positions=test_pos, bbox=np.array((-80.0, 20.0, -60.0, 40.0), float),
             m_tx=args.m, n_subcarriers=args.f, subcarrier_spacing=SUBCARRIER_SPACING,
             carrier_freq=FC, n_per_position=1, sampling="single_real_per_position")
    print(f"saved -> {te}  ({h_te.shape[0]} x {h_te.shape[1]})", flush=True)
    os._exit(0)   # drjit/mitsuba teardown segfaults harmlessly after save


if __name__ == "__main__":
    main()
