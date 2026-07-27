"""
main_optimal.py — Optimal Per-(Component, Dimension) Quantizer + Torch GMM Infrastructure

Combines:
  - main.py: torch-based GMM params, EM, MAP prediction, conditional transforms
  - toy_example.py: per-(c,i) Gaussian CDF entropy-matched quantizer design,
                     per-component conditional entropy for operational rate
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

from main import (
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


# ================= Optimal Quantizer Design (from toy_example.py, kept as numpy/scipy) =================

def _compute_normalized_entropy(delta_norm: float) -> float:
    """
    Compute entropy of uniform quantizer with step delta_norm for N(0,1).
    Since H only depends on delta/sigma, this is the universal building block.
    """
    K_max = int(6.0 / delta_norm) + 10

    probs = []
    for k in range(-K_max, K_max + 1):
        p_k = norm.cdf((k + 0.5) * delta_norm) - \
              norm.cdf((k - 0.5) * delta_norm)
        if p_k > 1e-10:
            probs.append(p_k)

    probs = np.array(probs)
    probs = probs / probs.sum()
    H = -np.sum(probs * np.log2(probs + 1e-30))
    return float(H)


# ---- Normalized lookup table: rate -> delta_norm (for sigma=1) ----
_NORM_RATES_ASC: Optional[np.ndarray] = None
_NORM_DELTAS_ASC: Optional[np.ndarray] = None
_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantizer_table_old.npz")


def _ensure_normalized_table(n_points: int = 2000):
    """Load lookup table from disk, or build and save if not found."""
    global _NORM_RATES_ASC, _NORM_DELTAS_ASC
    if _NORM_RATES_ASC is not None:
        return

    # Try loading from disk
    if os.path.exists(_TABLE_PATH):
        data = np.load(_TABLE_PATH)
        _NORM_RATES_ASC = data["rates"]
        _NORM_DELTAS_ASC = data["deltas"]
        print(f"  [Quantizer table] loaded from {os.path.basename(_TABLE_PATH)} "
              f"({len(_NORM_RATES_ASC)} pts, "
              f"rate range [{_NORM_RATES_ASC[0]:.3f}, {_NORM_RATES_ASC[-1]:.3f}])")
        return

    # Build and save
    t0 = time.time()
    delta_norms = np.logspace(np.log10(1e-6), np.log10(15.0), n_points)
    rates = np.array([_compute_normalized_entropy(d) for d in delta_norms])
    # rates is decreasing (small delta → high rate); reverse for np.interp
    _NORM_RATES_ASC = rates[::-1].copy()
    _NORM_DELTAS_ASC = delta_norms[::-1].copy()
    np.savez(_TABLE_PATH, rates=_NORM_RATES_ASC, deltas=_NORM_DELTAS_ASC)
    print(f"  [Quantizer table] built {n_points} pts in {time.time()-t0:.2f}s, "
          f"saved to {os.path.basename(_TABLE_PATH)}")


def lookup_delta(target_rate: float, sigma: float) -> float:
    """
    Fast quantizer step-size lookup.

    1. Look up normalized delta for target_rate from precomputed table (sigma=1)
    2. Scale by sigma to get actual delta

    Since H(delta, sigma) = H(delta/sigma, 1), the normalized table is universal.
    """
    _ensure_normalized_table()
    if target_rate < 0.01:
        return sigma * 10.0
    delta_norm = float(np.interp(target_rate, _NORM_RATES_ASC, _NORM_DELTAS_ASC))
    return delta_norm * sigma 


# ================= Per-(c,i) Quantizer Infrastructure =================

def precompute_quantizers(
    params: RealGMMParams,
    gamma: float,
) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    """
    Pre-compute per-(component, dimension) quantizers.

    For active dim (λ_{c,i} > γ):
        R_{c,i} = 0.5 * log2(λ_{c,i} / γ)
        delta_{c,i} = lookup_delta(R_{c,i}, sqrt(λ_{c,i}))

    Returns:
        quantizers: dict mapping (c, i) -> delta_{c,i}
        rate_table: (K, N) theoretical rate allocation
        active_table: (K, N) bool active mask
    """
    _ensure_normalized_table()
    lam = params.lam.detach().cpu()  # (K, N)
    K, N = lam.shape

    quantizers = {}
    rate_table = torch.zeros(K, N)
    active_table = torch.zeros(K, N, dtype=torch.bool)

    for c in range(K):
        for i in range(N):
            lam_ci = float(lam[c, i])
            if lam_ci > gamma:
                active_table[c, i] = True
                R_ci = 0.5 * math.log2(lam_ci / gamma)
                rate_table[c, i] = R_ci
                sigma_ci = math.sqrt(lam_ci)
                quantizers[(c, i)] = lookup_delta(R_ci, sigma_ci)
            else:
                active_table[c, i] = False
                rate_table[c, i] = 0.0
                quantizers[(c, i)] = 1.0  # dummy

    return quantizers, rate_table, active_table


@torch.no_grad()
def encode_with_precomputed_quantizers(
    y: torch.Tensor,          # (B, N)
    c_hat: torch.Tensor,      # (B,) long
    quantizers: dict,
    active_table: torch.Tensor,  # (K, N) bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode using per-(c,i) quantizers.

    Returns:
        q: (B, N) int64
        delta_map: (B, N) float — per-sample delta used
        active: (B, N) bool
    """
    B, N = y.shape
    device = y.device
    dtype = y.dtype

    q = torch.zeros(B, N, device=device, dtype=torch.int64)
    delta_map = torch.zeros(B, N, device=device, dtype=dtype)
    active = torch.zeros(B, N, device=device, dtype=torch.bool)

    K = active_table.shape[0]
    for c in range(K):
        mask_c = (c_hat == c)
        if not mask_c.any():
            continue
        idx_c = mask_c.nonzero(as_tuple=True)[0]

        for i in range(N):
            if active_table[c, i]:
                delta_ci = quantizers[(c, i)]
                delta_map[idx_c, i] = delta_ci
                active[idx_c, i] = True
                q[idx_c, i] = torch.round(y[idx_c, i] / delta_ci).to(torch.int64)

    return q, delta_map, active


@torch.no_grad()
def decode_with_precomputed_quantizers(
    q: torch.Tensor,          # (B, N) int64
    delta_map: torch.Tensor,  # (B, N) float
    active: torch.Tensor,     # (B, N) bool
    c_hat: torch.Tensor,      # (B,) long
    params: RealGMMParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode: dequantize + inverse transform.

    Returns:
        y_hat: (B, N)
        x_hat: (B, N)
    """
    dtype = delta_map.dtype
    y_hat = torch.zeros_like(q, dtype=dtype)
    if active.any():
        y_hat[active] = q[active].to(dtype) * delta_map[active]

    x_hat = conditional_inverse_transform(y_hat, c_hat, params)
    return y_hat, x_hat


# ================= Corrected Operational Rate =================

@torch.no_grad()
def empirical_entropy_bits(x: torch.Tensor) -> float:
    """Empirical entropy H(X) in bits for 1D integer tensor."""
    if x.numel() == 0:
        return 0.0
    vals, counts = torch.unique(x, return_counts=True)
    p = counts.to(torch.float64) / counts.sum().to(torch.float64)
    H = -(p * torch.log2(p)).sum().item()
    return float(H)


@torch.no_grad()
def operational_rate_conditioned(
    q: torch.Tensor,          # (B, N) int64
    active: torch.Tensor,     # (B, N) bool
    c_hat: torch.Tensor,      # (B,) long
    pi: torch.Tensor,         # (K,) mixture weights
    H_C: float,
) -> float:
    """
    Operational rate with per-component conditional entropy.

    R = H(C) + Σ_c π_c × Σ_i H(q_i | C=c)
    """
    K = pi.shape[0]
    B, N = q.shape
    pi_cpu = pi.detach().cpu()
    q_cpu = q.detach().cpu()
    c_cpu = c_hat.detach().cpu().to(torch.long)
    act_cpu = active.detach().cpu()

    R_idx = 0.0
    for c in range(K):
        mask_c = (c_cpu == c)
        if not mask_c.any():
            continue
        for i in range(N):
            mask_ci = mask_c & act_cpu[:, i]
            if mask_ci.any():
                q_ci = q_cpu[mask_ci, i]
                R_idx += float(pi_cpu[c]) * empirical_entropy_bits(q_ci)

    return H_C + R_idx


# ================= RD Sweep =================

@torch.no_grad()
def sweep_rd_optimal(
    x: torch.Tensor,              # (B, N)
    c_hat: torch.Tensor,          # (B,)
    params: RealGMMParams,
    gammas: torch.Tensor,          # (G,)
    H_label: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sweep gamma and compute theoretical + operational RD using optimal quantizers.

    Returns:
        R_th, D_th, R_op, D_op — each (G,) tensor
    """
    device = x.device
    dtype = x.dtype

    y, lam_sel = conditional_transform(x, c_hat, params)

    G = gammas.numel()
    R_th = torch.zeros(G)
    D_th = torch.zeros(G)
    R_op = torch.zeros(G)
    D_op = torch.zeros(G)

    t_total_start = time.time()
    t_theory = 0.0
    t_quantizer = 0.0
    t_encode = 0.0
    t_decode = 0.0
    t_rate = 0.0

    for gi, g in enumerate(gammas.tolist()):
        # Theoretical
        t0 = time.time()
        _, r_sum, _ = theoretical_rwf_rate(lam_sel, gamma=g)
        d_vec = theoretical_distortion_from_lam(lam_sel, gamma=g)
        R_th[gi] = r_sum.mean().item() + H_label
        D_th[gi] = d_vec.mean().item()
        t_theory += time.time() - t0

        # Operational: optimal per-(c,i) quantizers
        t0 = time.time()
        quantizers, rate_table, active_table = precompute_quantizers(params, g)
        t_quantizer += time.time() - t0

        t0 = time.time()
        q, delta_map, active = encode_with_precomputed_quantizers(y, c_hat, quantizers, active_table)
        t_encode += time.time() - t0

        t0 = time.time()
        _, x_hat = decode_with_precomputed_quantizers(q, delta_map, active, c_hat, params)
        t_decode += time.time() - t0

        t0 = time.time()
        D_op[gi] = mse_per_vector(x, x_hat).mean().item()
        R_op[gi] = operational_rate_conditioned(q, active, c_hat, params.pi, H_label)
        t_rate += time.time() - t0

        if (gi + 1) % 5 == 0 or gi == 0 or gi == G - 1:
            print(f"  [gamma {gi+1}/{G}] γ={g:.4f}, R_op={R_op[gi]:.2f}, D_op={D_op[gi]:.4f}")

    t_total = time.time() - t_total_start
    print(f"  [Timing] total={t_total:.2f}s | theory={t_theory:.2f}s, "
          f"quantizer={t_quantizer:.2f}s, encode={t_encode:.2f}s, "
          f"decode={t_decode:.2f}s, rate_calc={t_rate:.2f}s")

    return R_th, D_th, R_op, D_op


# ================= Main =================

def main():
    parser = argparse.ArgumentParser(description="Optimal Per-(c,i) Quantizer RD Simulation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=32, help="Source dimension")
    parser.add_argument("--K", type=int, default=4, help="Number of mixture components")
    parser.add_argument("--r", type=int, default=6, help="Number of dominant eigenvalues")
    parser.add_argument("--lam_hi", type=float, default=10.0)
    parser.add_argument("--lam_lo", type=float, default=0.1)
    parser.add_argument("--mean_scale", type=float, default=0.0, help="GMM mean separation scale")
    parser.add_argument("--T_train", type=int, default=20000)
    parser.add_argument("--T_test", type=int, default=5000)
    parser.add_argument("--use_em", action="store_true", help="Include EM-fitted curve")
    parser.add_argument("--em_iter", type=int, default=50)
    parser.add_argument("--em_eps", type=float, default=1e-6)
    parser.add_argument("--gamma_min", type=float, default=1e-3)
    parser.add_argument("--gamma_max", type=float, default=10.0)
    parser.add_argument("--num_gamma", type=int, default=25)
    parser.add_argument("--save", type=str, default="rd_optimal.png", help="Path to save plot")
    args = parser.parse_args()

    print(f"[Config] N={args.N}, K={args.K}, r={args.r}, mean_scale={args.mean_scale}")

    # 1. Build true GMM params
    params_true = build_real_gmm_params(
        N=args.N, K=args.K, r=args.r,
        lam_hi=args.lam_hi, lam_lo=args.lam_lo,
        mean_scale=args.mean_scale,
        seed=args.seed,
    )
    # Override eigenvalues: log-uniform random in [lam_lo, lam_hi], sorted descending
    torch.manual_seed(args.seed + 999)
    log_lo = math.log(args.lam_lo)
    log_hi = math.log(args.lam_hi)
    lam_rand = torch.exp(torch.rand(args.K, args.N) * (log_hi - log_lo) + log_lo)
    lam_rand, _ = lam_rand.sort(dim=1, descending=True)
    params_true = RealGMMParams(
        pi=params_true.pi, mu=params_true.mu,
        U=params_true.U, lam=lam_rand,
    )
    print(f"[GMM] pi={params_true.pi.numpy()}")
    print(f"[GMM] lam shape={tuple(params_true.lam.shape)}, "
          f"range=[{params_true.lam.min():.4f}, {params_true.lam.max():.4f}]")

    # 2. Sample train/test
    torch.manual_seed(args.seed)
    x_train, c_train = sample_real_gmm(args.T_train, params_true)
    x_test, c_test = sample_real_gmm(args.T_test, params_true)
    print(f"[Data] train={x_train.shape}, test={x_test.shape}")

    # 3. Oracle MAP prediction on test set
    c_hat_oracle, _ = predict_modes(x_test, params_true)
    acc_oracle = (c_hat_oracle == c_test).float().mean().item()
    print(f"[Oracle MAP] accuracy={acc_oracle:.4f}")

    # 4. Gamma sweep
    gammas = torch.logspace(
        math.log10(args.gamma_min),
        math.log10(args.gamma_max),
        steps=args.num_gamma,
    )
    H_pi_true = label_entropy_from_pi(params_true.pi)
    print(f"[H(C)] = {H_pi_true:.4f} bits")

    curves = []  # list of (name, R, D, style)

    # (a) Theoretical RD bound (using true labels)
    R_th, D_th, _, _ = sweep_rd_optimal(x_test, c_test, params_true, gammas, H_pi_true)
    curves.append(("Theoretical RD bound", R_th, D_th, "b-"))

    # (b) Oracle params + true labels + optimal quantizer
    _, _, R_op_true, D_op_true = sweep_rd_optimal(x_test, c_test, params_true, gammas, H_pi_true)
    curves.append(("Oracle params + True labels", R_op_true, D_op_true, "g--"))

    # (c) Oracle params + MAP labels + optimal quantizer
    _, _, R_op_map, D_op_map = sweep_rd_optimal(x_test, c_hat_oracle, params_true, gammas, H_pi_true)
    curves.append(("Oracle params + MAP labels", R_op_map, D_op_map, "m--"))

    # (d) EM params + MAP labels + optimal quantizer
    if args.use_em:
        print(f"\n[EM] Fitting on {min(args.T_train, len(x_train))} training samples...")
        x_em = x_train[:min(args.T_train, len(x_train))]
        params_em = gmm_em_fit_fullcov(
            x_em, K=args.K,
            num_iter=args.em_iter, eps=args.em_eps,
            seed=args.seed, verbose=True,
        )
        print(f"[EM] pi_hat={params_em.pi.detach().cpu().numpy()}")

        # MAP with EM params (labels may be permuted)
        c_hat_em, _ = predict_modes(x_test, params_em)
        best_acc, best_perm, c_hat_em_mapped = best_label_permutation(c_test, c_hat_em, K=args.K)
        print(f"[EM MAP] best-perm accuracy={best_acc:.4f}, perm={best_perm}")

        H_pi_em = label_entropy_from_pi(params_em.pi)

        # (d-1) EM params + True labels (map true labels to EM label space)
        inv_perm = [0] * args.K
        for em_idx, true_idx in enumerate(best_perm):
            inv_perm[true_idx] = em_idx
        c_test_em_space = torch.tensor([inv_perm[c.item()] for c in c_test])
        print(f"[EM] inv_perm (true→EM): {inv_perm}")

        _, _, R_op_em_true, D_op_em_true = sweep_rd_optimal(
            x_test, c_test_em_space, params_em, gammas, H_pi_em)
        curves.append(("EM params + True labels", R_op_em_true, D_op_em_true, "c-."))

        # (d-2) EM params + MAP labels
        _, _, R_op_em, D_op_em = sweep_rd_optimal(x_test, c_hat_em, params_em, gammas, H_pi_em)
        curves.append(("EM params + MAP labels", R_op_em, D_op_em, "r:"))

    # (e) Single Gaussian baseline (no label overhead)
    print(f"\n[Single Gaussian] Fitting on training data...")
    sg_params = fit_single_gaussian_fullcov(x_train)
    # Wrap into RealGMMParams with K=1 so we can reuse sweep_rd_optimal
    params_sg = RealGMMParams(
        pi=torch.tensor([1.0]),
        mu=sg_params.mu.unsqueeze(0),      # (1, N)
        U=sg_params.U.unsqueeze(0),        # (1, N, N)
        lam=sg_params.lam.unsqueeze(0),    # (1, N)
    )
    c_sg = torch.zeros(x_test.shape[0], dtype=torch.long)  # all label=0
    H_sg = 0.0  # no label entropy
    print(f"[Single Gaussian] H(C)=0, lam range=[{sg_params.lam.min():.4f}, {sg_params.lam.max():.4f}]")
    _, _, R_op_sg, D_op_sg = sweep_rd_optimal(x_test, c_sg, params_sg, gammas, H_sg)
    curves.append(("Single Gaussian (no label)", R_op_sg, D_op_sg, "k-."))

    # 5. Plot
    pixels_per_vector = args.N
    MAX = 1.0
    eps = 1e-12

    plt.figure(figsize=(10, 7))
    for name, R, D, style in curves:
        bpp = (R / pixels_per_vector).numpy()
        D_np = D.numpy()
        psnr = 10 * np.log10((MAX**2) / (D_np + eps))

        order = np.argsort(bpp)
        plt.plot(bpp[order], psnr[order], style, linewidth=2, label=name)

    plt.xlabel("Rate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Optimal Quantizer RD Curves (N={args.N}, K={args.K}, r={args.r})")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    if args.save:
        plt.savefig(args.save, bbox_inches="tight", dpi=200)
        print(f"\n[Saved] {args.save}")
    else:
        plt.show()

    print("\n[Done]")


if __name__ == "__main__":
    main()
