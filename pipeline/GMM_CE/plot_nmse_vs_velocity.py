"""NMSE vs. velocity (full pilot / comb 8) 두 panel -> nmse_vs_velocity_paper.pdf.
기존 results_subregion_tf_v{v}_ulatf[_comb8]_paper50/nmse_vs_snr.json 에서 SNR=30 dB 값 사용
(해당 json 은 K=128, N_train=1e5).  재실험 없음."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from replot_nmse_vs_snr import STYLE   # 폰트 rcParams 도 함께 적용됨

VEL, SNR = [0, 3, 6, 10], 30
KEYS = ["GMM full", "GMM b-circ", "GMM 2x1D-toep", "GMM kron", "GMM 2x1D", "GMM 2x1D-circ",
        "PDP+DS 2x1D", "PDP+DS kron"]
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, tag, ttl in zip(axs, ["", "_comb8"], ["Full pilot", "Comb 8"]):
    ds = [json.load(open(f"results_subregion_tf_v{v}_ulatf{tag}_paper50/nmse_vs_snr.json")) for v in VEL]
    for key, lab, c, m, ls in STYLE:
        if key not in KEYS: continue
        y = [10 * np.log10(d[key][d["snr_db"].index(SNR)]) for d in ds]
        ax.plot(VEL, y, label=lab, color=c, marker=m, ls=ls)
    ax.set_xticks(VEL); ax.set_xlabel("Velocity [m/s]"); ax.set_ylabel("NMSE [dB]")
    ax.set_title(f"{ttl} (K=128, SNR=30 dB, $N_{{train}}=10^5$)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
fig.tight_layout()
fig.savefig("nmse_vs_velocity_paper.pdf", bbox_inches="tight")
fig.savefig("/tmp/claude-1007/-home-ejseo/e497a807-299c-469a-b8f3-a29fa2fce350/scratchpad/vel.png", dpi=60, bbox_inches="tight")
