"""Matplotlib reproductions of Fig. 2 / 3 / 4 of the paper."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch


def _decade_ticks_only(ax, axis: str = "y"):
    """Show/label only power-of-ten ticks (10^0, 10^-1, ...) on a log
    axis, dropping the in-between minor ticks matplotlib otherwise adds
    when the plotted range spans only one or two decades."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,)))  # subs=(1.0,) so a <2-decade range doesn't auto-add 2x/3x/... ticks
    a.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    a.set_minor_locator(mticker.NullLocator())
    a.set_minor_formatter(mticker.NullFormatter())

STYLE = {
    "GMM full": dict(color="tab:blue", marker="s", linestyle="-"),
    "GMM kron": dict(color="black", marker="s", linestyle=":", markerfacecolor="none"),
    "GMM b-toep": dict(color="olive", marker="^", linestyle="-"),
    "GMM b-circ": dict(color="red", marker="^", linestyle="-."),
    "GMM 2x1D": dict(color="tab:purple", marker="d", linestyle="-"),
    "GMM 2x1D-toep": dict(color="olive", marker="^", linestyle="--"),
    "GMM 2x1D-circ": dict(color="red", marker="d", linestyle="-."),
    "PDP+DS kron": dict(color="black", marker="o", linestyle=":"),
    "PDP+DS 2x1D": dict(color="gray", marker="o", linestyle="-"),
}

# Legend reading order (row-major, 3 columns), matching the paper's Fig. 3
# and Fig. 4 legend layouts exactly (they differ in where 2x1D-toep/-circ
# and the PDP+DS pair fall).
FIG3_ORDER = ["GMM full", "GMM kron", "GMM b-toep",
              "GMM b-circ", "GMM 2x1D", "GMM 2x1D-circ",
              "GMM 2x1D-toep", "PDP+DS 2x1D", "PDP+DS kron"]
FIG4_ORDER = ["GMM full", "GMM kron", "GMM b-toep",
              "GMM b-circ", "GMM 2x1D", "GMM 2x1D-toep",
              "GMM 2x1D-circ"]

# Fig. 2 (bar chart) uses its own paper-matched palette/order: light green
# hatched ("GMM 2x1D"), white/light-gray hatched ("GMM kron"), solid light
# blue ("GMM full"), left-to-right per SNR group.
FIG2_STYLE = {
    "GMM 2x1D": dict(color="yellowgreen"),
    "GMM kron": dict(color="black"),
    "GMM full": dict(color="tab:blue"),
}


def plot_fig2(result: dict, save_path: str):
    """Stem plot, mirroring the paper's Fig. 2 (grouped bars on a log
    y-axis, one group of GMM full / kron / 2x1D stems per SNR value,
    y-axis range 10^0-10^2 GMM components)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["GMM 2x1D", "GMM kron", "GMM full"]  # left-to-right order, matching the paper
    offsets = [-1.5, 0.0, 1.5]  # dB, so the three (thicker) stems per SNR don't overlap
    bar_width = 8  # points; thick enough to read as a bar rather than a thin stem
    for name, off in zip(names, offsets):
        style = FIG2_STYLE[name]
        xs = [s + off for s in result["snr_db"]]
        markerline, stemlines, baseline = ax.stem(xs, result[name], basefmt=" ")
        markerline.set_marker("")  # no marker dot at the bar top
        stemlines.set_color(style["color"])
        stemlines.set_linewidth(bar_width)
        stemlines.set_capstyle("butt")  # flat bar top/bottom, not rounded
    ax.set_yscale("log")
    ax.set_ylim(1e0, 1e2)
    _decade_ticks_only(ax)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("GMM components (avg. for 99% responsibility)")
    ax.set_title("Fig. 2 reproduction (synthetic channel)")
    ax.grid(True, which="major", alpha=0.3)
    legend_handles = [Patch(facecolor=FIG2_STYLE[n]["color"], edgecolor="black", label=n) for n in names]
    ax.legend(handles=legend_handles)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig3(result_low_v: dict, result_high_v: dict, save_path: str):
    # y-axis range is left dynamic (autoscaled to the data) rather than
    # pinned to the paper's fixed 10^-4/10^-3..10^0, so curves that now
    # dip below what the paper's axis showed don't get clipped.
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for ax, result, title in zip(axes, [result_low_v, result_high_v],
                                  ["v = 3 km/h", "v in [0, 300] km/h"]):
        for name in FIG3_ORDER:
            if name in result:
                ax.plot(result["snr_db"], result[name], label=name, **STYLE[name])
        ax.set_yscale("log")
        _decade_ticks_only(ax)
        ax.set_ylabel("Normalized MSE")
        ax.set_title(title)
        ax.grid(True, which="major", alpha=0.3)
    axes[-1].set_xlabel("SNR [dB]")
    axes[0].legend(fontsize=8, ncol=3)
    fig.suptitle("Fig. 3 reproduction (synthetic channel -- see README)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig4(result_train: dict, result_K: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in FIG4_ORDER:
        if name in result_train:
            axes[0].plot(result_train["n_train"], result_train[name], label=name, **STYLE[name])
        if name in result_K:
            axes[1].plot(result_K["K"], result_K[name], label=name, **STYLE[name])
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(1e-2, 1e-1)  # matches the paper's Fig. 4 y-axis range
        _decade_ticks_only(ax)
        ax.set_ylabel("Normalized MSE")
        ax.grid(True, which="major", alpha=0.3)
    axes[0].set_xlabel("Training data")
    axes[1].set_xlabel("GMM components")
    axes[0].legend(fontsize=8, ncol=3)
    fig.suptitle("Fig. 4 reproduction (synthetic channel -- see README)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
