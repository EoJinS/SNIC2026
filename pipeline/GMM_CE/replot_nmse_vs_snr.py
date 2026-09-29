"""기존 results_subregion_*/nmse_vs_snr.json 을 example_nmse.png 양식(plot_nmse_vs_snr.py)으로 다시 그린다.
실험 재실행 없음. 출력: <dir>/nmse_vs_snr_paper.{png,pdf}.  LS 는 NMSE=1/SNR 로 해석적 계산.
사용: python replot_nmse_vs_snr.py [--with-pdp] dir1 dir2 ...
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = 2
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    "font.size": 10 + B, "axes.labelsize": 10 + B, "xtick.labelsize": 10 + B,
    "ytick.labelsize": 10 + B, "legend.fontsize": 10 + B, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# json 키 -> (범례 라벨, 색, 마커, ls)
STYLE = [
    ("GMM full",      r"GMM full",               "tab:blue",   "s", "-"),
    ("GMM b-toep",    r"GMM b-toep",             "tab:cyan",   "<", "-"),
    ("GMM b-circ",    r"GMM b-circ",             "tab:red",    "^", "-"),
    ("GMM kron",      r"GMM kron",               "k",          "o", "-"),
    ("GMM 2x1D",      r"GMM 2$\times$1D",        "tab:purple", "d", "-"),
    ("GMM 2x1D-toep", r"GMM 2$\times$1D-toep",   "tab:green",  "v", "-"),
    ("GMM 2x1D-circ", r"GMM 2$\times$1D-circ",   "tab:orange", "x", "-"),
    ("GMM Weichselberger", r"GMM Weichselberger", "tab:brown", "P", "-"),
    ("PDP+DS 2x1D",   r"PDP+DS 2$\times$1D",     "tab:pink",   "*", "-"),
    ("PDP+DS kron",   r"PDP+DS kron",            "tab:olive",  "D", "-"),
]

def plot(d, with_pdp):
    snr = np.array(d["snr_db"], float)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for key, lab, c, m, ls in STYLE:
        if key not in d or (key.startswith("PDP") and not with_pdp):
            continue
        ax.plot(snr, 10 * np.log10(d[key]), label=lab, color=c, marker=m, ls=ls)
    ax.plot(snr, -snr, label="LS", color="0.5", marker=".", ls="--")
    ax.set_xlabel("SNR [dB]"); ax.set_ylabel("NMSE [dB]")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout()
    return fig

def plot_triple(dr, with_pdp):
    """components / snr / training 세 panel 을 가로 일렬로 (PDF 한 장)."""
    base = ["GMM full", "GMM b-circ", "GMM 2x1D-toep", "GMM kron", "GMM 2x1D", "GMM 2x1D-circ",
            "GMM Weichselberger"]
    snr_keys = base + ["PDP+DS 2x1D", "PDP+DS kron"]
    panels = [("components", "K", "GMM components", True, base), ("snr", "snr_db", "SNR [dB]", False, snr_keys),
              ("training", "n_train", "Training data", True, base)]
    fig, axs = plt.subplots(1, 3, figsize=(21, 5.5))
    for ax, (name, xk, xl, logx, keys) in zip(axs, panels):
        d = json.load(open(os.path.join(dr, f"nmse_vs_{name}.json")))
        x = np.array(d[xk], float)
        for key, lab, c, m, ls in STYLE:
            if key not in keys or key not in d:
                continue
            ax.plot(x, 10 * np.log10(d[key]), label=lab, color=c, marker=m, ls=ls)
        if logx:
            ax.set_xscale("log"); ax.set_xticks(x)
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel(xl); ax.set_ylabel("NMSE [dB]")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--triple", action="store_true", help="components/snr/training 3-panel PDF")
    p.add_argument("--with-pdp", action="store_true")
    a = p.parse_args()
    for dr in a.dirs:
        if a.triple:
            fig = plot_triple(dr, a.with_pdp)
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(dr, f"nmse_triple_paper.{ext}"), dpi=150, bbox_inches="tight")
            plt.close(fig); print("saved", dr); continue
        d = json.load(open(os.path.join(dr, "nmse_vs_snr.json")))
        fig = plot(d, a.with_pdp)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(dr, f"nmse_vs_snr_paper.{ext}"), dpi=150, bbox_inches="tight")
        plt.close(fig); print("saved", dr)
