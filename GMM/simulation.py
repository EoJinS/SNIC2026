"""
main_optimal_custom_dataset.py — Optimal Quantizer RD for Custom or Synthetic Datasets

Default: Load custom dataset (--data), fit GMM via EM, compress with optimal quantizer.
With --use_synthetic_data: Generate synthetic GMM data with oracle params (same as main_optimal.py).

Compression pipeline is identical to main_optimal.py:
  - Per-(c,i) Gaussian CDF entropy-matched quantizer
  - Per-component conditional entropy for operational rate
"""

import os
import math
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from scipy.stats import norm

from SNIC2026.GMM.main import (
    RealGMMParams,
    build_real_gmm_params,
    sample_real_gmm,
    gmm_em_fit_fullcov,
    predict_modes,
    conditional_transform,
    conditional_inverse_transform,
    label_entropy_from_pi,
    best_label_permutation,
    mse_per_vector,
    theoretical_rwf_rate,
    theoretical_distortion_from_lam,
    fit_single_gaussian_fullcov,
    single_gaussian_transform,
    single_gaussian_inverse_transform,
)

# ================= Reuse quantizer infrastructure from main_optimal =================
from main_optimal import (
    _ensure_normalized_table,
    lookup_delta,
    precompute_quantizers,
    encode_with_precomputed_quantizers,
    decode_with_precomputed_quantizers,
    empirical_entropy_bits,
    operational_rate_conditioned,
    sweep_rd_optimal,
)


# ================= Data Loading =================

def load_custom_dataset(
    path: str,
    train_ratio: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load custom dataset from file.

    Supported formats:
      - .pt / .pth: torch tensor (T, N)
      - .npy: numpy array (T, N)
      - .npz: numpy archive with key 'data' or first array

    Returns:
        x_train: (T_train, N)
        x_test: (T_test, N)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.pt', '.pth'):
        data = torch.load(path, weights_only=True)
        if isinstance(data, dict):
            # Try common keys
            for key in ('data', 'x', 'X', 'train'):
                if key in data:
                    data = data[key]
                    break
            else:
                raise ValueError(f"Dict keys: {list(data.keys())}. Need 'data', 'x', 'X', or 'train'.")
        x = data.to(torch.float32)
    elif ext == '.npy':
        x = torch.from_numpy(np.load(path)).to(torch.float32)
    elif ext == '.npz':
        npz = np.load(path)
        keys = list(npz.keys())
        for key in ('data', 'x', 'X', 'train'):
            if key in keys:
                x = torch.from_numpy(npz[key]).to(torch.float32)
                break
        else:
            x = torch.from_numpy(npz[keys[0]]).to(torch.float32)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .pt, .npy, or .npz")

    if x.dim() != 2:
        raise ValueError(f"Expected 2D data (T, N), got shape {tuple(x.shape)}")

    T = x.shape[0]
    T_train = int(T * train_ratio)
    if T_train < 100 or (T - T_train) < 50:
        raise ValueError(f"Dataset too small: {T} samples. Need at least ~200.")

    perm = torch.randperm(T)
    x_train = x[perm[:T_train]]
    x_test = x[perm[T_train:]]

    return x_train, x_test


# ================= Main =================

def main():
    parser = argparse.ArgumentParser(
        description="Optimal Quantizer RD — Custom Dataset or Synthetic")

    # --- Mode ---
    parser.add_argument("--use_synthetic_data", action="store_true",
                        help="Use synthetic GMM data (with oracle curves)")
    parser.add_argument("--data", type=str, default="",
                        help="Path to custom dataset (.pt, .npy, .npz)")

    # --- Common ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=4, help="Number of GMM components for EM")
    parser.add_argument("--em_iter", type=int, default=50)
    parser.add_argument("--em_eps", type=float, default=1e-6)
    parser.add_argument("--em_tol", type=float, default=1e-3,
                        help="Early stopping: stop if |dLL/LL| < tol")
    parser.add_argument("--gamma_min", type=float, default=1e-3)
    parser.add_argument("--gamma_max", type=float, default=10.0)
    parser.add_argument("--num_gamma", type=int, default=25)
    parser.add_argument("--save", type=str, default="rd_custom.png",
                        help="Path to save plot")
    parser.add_argument("--train_ratio", type=float, default=0.8)

    # --- Synthetic-only ---
    parser.add_argument("--N", type=int, default=32, help="Dimension (synthetic)")
    parser.add_argument("--r", type=int, default=6, help="Dominant eigenvalues (synthetic)")
    parser.add_argument("--lam_hi", type=float, default=10.0)
    parser.add_argument("--lam_lo", type=float, default=0.1)
    parser.add_argument("--mean_scale", type=float, default=0.0)
    parser.add_argument("--T_train", type=int, default=20000)
    parser.add_argument("--T_test", type=int, default=5000)

    args = parser.parse_args()

    # ============================================================
    # 1. Load / Generate data
    # ============================================================
    if args.use_synthetic_data:
        print(f"[Mode] Synthetic data (N={args.N}, K={args.K})")
        params_true = build_real_gmm_params(
            N=args.N, K=args.K, r=args.r,
            lam_hi=args.lam_hi, lam_lo=args.lam_lo,
            mean_scale=args.mean_scale, seed=args.seed,
        )
        # Override eigenvalues: log-uniform random
        torch.manual_seed(args.seed + 999)
        log_lo, log_hi = math.log(args.lam_lo), math.log(args.lam_hi)
        lam_rand = torch.exp(torch.rand(args.K, args.N) * (log_hi - log_lo) + log_lo)
        lam_rand, _ = lam_rand.sort(dim=1, descending=True)
        params_true = RealGMMParams(
            pi=params_true.pi, mu=params_true.mu,
            U=params_true.U, lam=lam_rand,
        )

        torch.manual_seed(args.seed)
        x_train, c_train = sample_real_gmm(args.T_train, params_true)
        x_test, c_test = sample_real_gmm(args.T_test, params_true)
        N = args.N
        has_oracle = True
        print(f"[GMM] lam range=[{params_true.lam.min():.4f}, {params_true.lam.max():.4f}]")
        print(f"[Data] train={x_train.shape}, test={x_test.shape}")
    else:
        if not args.data:
            parser.error("--data is required unless --use_synthetic_data is set")
        print(f"[Mode] Custom dataset: {args.data}")
        torch.manual_seed(args.seed)
        x_train, x_test = load_custom_dataset(args.data, args.train_ratio)
        N = x_train.shape[1]
        has_oracle = False
        params_true = None
        c_train = None
        c_test = None
        print(f"[Data] train={x_train.shape}, test={x_test.shape}, N={N}")

    # ============================================================
    # 2. EM fitting (always run)
    # ============================================================
    print(f"\n[EM] Fitting K={args.K} on {len(x_train)} training samples "
          f"(max {args.em_iter} iters, tol={args.em_tol})...")
    t_em_start = time.time()
    params_em = gmm_em_fit_fullcov(
        x_train, K=args.K,
        num_iter=args.em_iter, eps=args.em_eps,
        seed=args.seed, verbose=True,
    )
    t_em = time.time() - t_em_start
    print(f"[EM] Done in {t_em:.2f}s")
    print(f"[EM] pi_hat={params_em.pi.detach().cpu().numpy()}")
    print(f"[EM] lam range=[{params_em.lam.min():.4f}, {params_em.lam.max():.4f}]")

    # MAP classification with EM params
    c_hat_em, _ = predict_modes(x_test, params_em)
    H_pi_em = label_entropy_from_pi(params_em.pi)
    print(f"[EM] H(C)={H_pi_em:.4f} bits")

    # ============================================================
    # 3. Gamma sweep
    # ============================================================
    gammas = torch.logspace(
        math.log10(args.gamma_min),
        math.log10(args.gamma_max),
        steps=args.num_gamma,
    )

    curves = []  # (name, R, D, style)

    # --- Oracle curves (synthetic only) ---
    if has_oracle:
        H_pi_true = label_entropy_from_pi(params_true.pi)

        # Oracle MAP
        c_hat_oracle, _ = predict_modes(x_test, params_true)
        acc_oracle = (c_hat_oracle == c_test).float().mean().item()
        print(f"\n[Oracle MAP] accuracy={acc_oracle:.4f}")

        # (a) Theoretical RD bound
        print("\n--- Theoretical RD bound ---")
        R_th, D_th, _, _ = sweep_rd_optimal(
            x_test, c_test, params_true, gammas, H_pi_true)
        curves.append(("Theoretical RD bound", R_th, D_th, "b-"))

        # (b) Oracle params + True labels
        print("\n--- Oracle params + True labels ---")
        _, _, R_op_true, D_op_true = sweep_rd_optimal(
            x_test, c_test, params_true, gammas, H_pi_true)
        curves.append(("Oracle + True labels", R_op_true, D_op_true, "g--"))

        # (c) Oracle params + MAP labels
        print("\n--- Oracle params + MAP labels ---")
        _, _, R_op_map, D_op_map = sweep_rd_optimal(
            x_test, c_hat_oracle, params_true, gammas, H_pi_true)
        curves.append(("Oracle + MAP labels", R_op_map, D_op_map, "m--"))

        # EM label permutation alignment
        best_acc, best_perm, _ = best_label_permutation(c_test, c_hat_em, K=args.K)
        print(f"[EM MAP] best-perm accuracy={best_acc:.4f}, perm={best_perm}")

    # --- EM curves (always) ---
    print("\n--- EM params + MAP labels ---")
    R_th_em, D_th_em, R_op_em, D_op_em = sweep_rd_optimal(
        x_test, c_hat_em, params_em, gammas, H_pi_em)
    curves.append(("Theoretical RD (EM params)", R_th_em, D_th_em, "b-"))
    curves.append(("EM + MAP labels", R_op_em, D_op_em, "r:"))

    # --- Single Gaussian baseline (always) ---
    print(f"\n--- Single Gaussian baseline ---")
    sg_params = fit_single_gaussian_fullcov(x_train)
    params_sg = RealGMMParams(
        pi=torch.tensor([1.0]),
        mu=sg_params.mu.unsqueeze(0),
        U=sg_params.U.unsqueeze(0),
        lam=sg_params.lam.unsqueeze(0),
    )
    c_sg = torch.zeros(x_test.shape[0], dtype=torch.long)
    _, _, R_op_sg, D_op_sg = sweep_rd_optimal(
        x_test, c_sg, params_sg, gammas, 0.0)
    curves.append(("Single Gaussian (no label)", R_op_sg, D_op_sg, "k-."))

    # ============================================================
    # 4. Plot
    # ============================================================
    pixels_per_vector = N
    
    # Calculate normalization factor: average power per vector E[||x||^2]
    x_test_power = torch.mean(torch.sum(x_test**2, dim=1)).item()
    eps = 1e-12

    plt.figure(figsize=(10, 7))
    for name, R, D, style in curves:
        bpp = (R / pixels_per_vector).numpy()
        mse = D.numpy()
        nmse = mse / (x_test_power + eps)

        order = np.argsort(bpp)
        plt.plot(bpp[order], nmse[order], style, linewidth=2, label=name)

    mode_str = "Synthetic" if has_oracle else os.path.basename(args.data)
    plt.xlabel("Rate (bpp)")
    plt.ylabel("NMSE (dB or linear scale)")
    plt.yscale("log")
    plt.title(f"RD Curves (NMSE) — {mode_str} (N={N}, K={args.K})")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc="best")

    if args.save:
        plt.savefig(args.save, bbox_inches="tight", dpi=200)
        print(f"\n[Saved] {args.save}")
    else:
        plt.show()

    print("\n[Done]")


if __name__ == "__main__":
    main()
