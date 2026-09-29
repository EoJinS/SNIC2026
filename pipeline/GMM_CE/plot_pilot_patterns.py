"""Draw the two pilot lattices used by run_subregion_tf_pilot_experiments.py
(carrier x OFDM-symbol grid, Nc=64, Nt=14): uniform comb vs circular comb.

    python3 plot_pilot_patterns.py            # -> pilot_patterns.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_subregion_tf_pilot_experiments import make_pilot_grid


def draw(ax, grid, title, Nc, Nt):
    pil = np.zeros((Nt, Nc), bool)
    for r in range(grid.Np):
        idx = int(np.argmax(grid.A[r]))
        pil[idx // Nc, idx % Nc] = True
    tt, cc = np.nonzero(~pil)
    ax.scatter(cc + 1, tt + 1, s=10, facecolors="none", edgecolors="0.75", linewidths=0.6, label="Data (unobserved)")
    tt, cc = np.nonzero(pil)
    ax.scatter(cc + 1, tt + 1, s=42, marker="s", color="tab:blue", edgecolors="k", linewidths=0.8,
               label=f"Pilot (Np={grid.Np}/{Nc * Nt})")
    for t in grid.pilot_symbols:                       # first pilot carrier of each pilot symbol
        first = int(np.nonzero(pil[t])[0][0])
        ax.text(Nc + 1.5, t + 1, f"offset {first}", va="center", fontsize=8, color="tab:blue")
    ax.set_xlim(0, Nc + 8)
    ax.set_ylim(0.3, Nt + 0.7)
    ax.set_xticks([1, 9, 17, 25, 33, 41, 49, 57, 64])
    ax.set_yticks(range(1, Nt + 1))
    ax.invert_yaxis()
    ax.set_xlabel("Carrier (frequency)  Nc = 64")
    ax.set_ylabel("OFDM symbol (time)  Nt = 14")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot_patterns.png")
    ap.add_argument("--carrier-spacing", type=int, default=8)
    ap.add_argument("--time-spacing", type=int, default=2)
    ap.add_argument("--shift", type=int, default=1)
    a = ap.parse_args()
    Nc, Nt = 64, 14
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    u = make_pilot_grid(Nc, Nt, a.carrier_spacing, a.time_spacing, False)
    c = make_pilot_grid(Nc, Nt, a.carrier_spacing, a.time_spacing, True, a.shift)
    draw(axes[0], u, f"Uniform-uniform comb: carrier spacing {a.carrier_spacing}, symbol spacing {a.time_spacing} "
                      f"(same 8 carriers on every pilot symbol)", Nc, Nt)
    draw(axes[1], c, f"Circular-uniform comb: carrier spacing {a.carrier_spacing} shifted +{a.shift} carrier per "
                      f"pilot symbol, symbol spacing {a.time_spacing}", Nc, Nt)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(a.out, dpi=150)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
