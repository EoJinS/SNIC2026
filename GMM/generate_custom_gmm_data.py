"""
Generate a custom GMM dataset and save as .npy for main_optimal_custom_dataset.py.

Usage:
    python generate_custom_gmm_data.py
    python generate_custom_gmm_data.py --N 16 --K 3 --T 30000 --out my_data.npy
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate custom GMM dataset")
    parser.add_argument("--N", type=int, default=32, help="Dimension")
    parser.add_argument("--K", type=int, default=4, help="Number of components")
    parser.add_argument("--T", type=int, default=25000, help="Total samples")
    parser.add_argument("--mean_scale", type=float, default=2.0,
                        help="Scale of component means")
    parser.add_argument("--lam_lo", type=float, default=0.1)
    parser.add_argument("--lam_hi", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="custom_gmm_data.npy")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    N, K = args.N, args.K

    # --- Mixing weights ---
    alpha = rng.dirichlet(np.ones(K) * 2.0)  # mildly non-uniform
    print(f"pi = {alpha}")

    # --- Per-component: random mean, random covariance ---
    means = []
    covs = []
    for c in range(K):
        # Mean: random direction scaled
        mu_c = rng.standard_normal(N) * args.mean_scale
        means.append(mu_c)

        # Covariance: random orthogonal * diag(lam) * orthogonal^T
        # Log-uniform eigenvalues
        log_lo, log_hi = np.log(args.lam_lo), np.log(args.lam_hi)
        lam_c = np.exp(rng.uniform(log_lo, log_hi, size=N))
        lam_c = np.sort(lam_c)[::-1]  # descending

        # Random orthogonal matrix via QR
        A = rng.standard_normal((N, N))
        Q, _ = np.linalg.qr(A)
        Sigma_c = Q @ np.diag(lam_c) @ Q.T

        covs.append(Sigma_c)
        print(f"  Component {c}: lam range=[{lam_c[-1]:.4f}, {lam_c[0]:.4f}]")

    # --- Sample ---
    counts = rng.multinomial(args.T, alpha)
    print(f"Counts per component: {counts}")

    samples = []
    for c in range(K):
        x_c = rng.multivariate_normal(means[c], covs[c], size=counts[c])
        samples.append(x_c)

    X = np.concatenate(samples, axis=0).astype(np.float32)
    # Shuffle
    perm = rng.permutation(X.shape[0])
    X = X[perm]

    print(f"\nDataset shape: {X.shape}")
    print(f"Saving to: {args.out}")
    np.save(args.out, X)
    print("Done!")


if __name__ == "__main__":
    main()
