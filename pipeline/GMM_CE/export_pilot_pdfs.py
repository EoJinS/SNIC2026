"""Paper-style 3-panel figure (NMSE vs GMM components | SNR | training data, NMSE in dB) for
every pilot-pattern experiment directory. The drawing itself is ``replot_nmse_vs_snr.plot_triple``
(same style as results_subregion_tf_v0_ulatf_paper50/nmse_triple_paper.pdf); this script only
loops over the ``results_subregion_tf_v*_ulatf_{unif,circ}_c8t2_paper50/`` directories.

    python3 export_pilot_pdfs.py                      # every finished pilot-pattern experiment
    python3 export_pilot_pdfs.py results_subregion_tf_v6_ulatf_circ_c8t2_paper50

Writes <dir>/nmse_vs_snr_training_components.pdf (+ .png). Directories missing any of the three
JSONs are skipped (re-run once that experiment has finished). Estimators drawn = whatever is in
the JSON (unif: full set incl. PDP+DS on the SNR panel; circ: full / kron / b-circ), b-toep never.
"""
import glob
import os
import sys

import matplotlib.pyplot as plt

from replot_nmse_vs_snr import plot_triple

NAME = "nmse_vs_snr_training_components"


def export(d):
    need = [os.path.join(d, f"nmse_vs_{k}.json") for k in ("snr", "training", "components")]
    if not all(os.path.exists(f) for f in need):
        print(f"skip  {d}  (needs nmse_vs_snr/training/components.json)")
        return False
    fig = plot_triple(d, True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{NAME}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {os.path.join(d, NAME)}.pdf/.png")
    return True


if __name__ == "__main__":
    dirs = sys.argv[1:] or sorted(glob.glob("results_subregion_tf_v*_ulatf_*_c8t2_paper50"))
    done = sum(export(d) for d in dirs)
    print(f"{done}/{len(dirs)} figures written")
