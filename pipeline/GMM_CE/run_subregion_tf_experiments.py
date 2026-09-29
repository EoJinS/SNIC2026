"""
Run the GMM_CE structured-covariance experiments (Fesl et al., Asilomar
2022) on POSITION-ONLY ray-traced POSTECH ``subregion100`` channels laid
out on Fesl's OWN axes -- SISO, F subcarriers x S OFDM symbols per
position (a fixed UE speed, random heading) -- instead of
``run_subregion_experiments.py``'s (subcarrier, TX-antenna) proxy. That
MISO space-frequency script and its data (``gmmce/subregion_data.py``,
``results_subregion_M{m}*/``) are untouched; this is an additive sibling
sharing the same estimator library (``gmmce/pipeline.py`` et al., which is
axis-agnostic -- it only ever sees generic ``(Nc, Nt)``).

Axis mapping (see ``gmmce/subregion_tf_data.py``):
    Nc <- subcarrier / frequency   (delay-stationary   -> Toeplitz)
    Nt <- OFDM symbol / time       (Doppler-stationary  -> Toeplitz)
    C = kron(C_time, C_freq)   <->   exactly the paper's own kron ordering

Data: ``generate_channels_subregion100_siso_tf.py`` (S=14 symbols, F=64
subcarriers, v=10 m/s, D=896). Unlike the MISO driver there is still a
single NMSE-vs-SNR curve set (no separate v-sweep -- one representative
mobility operating point, see that script's docstring), but the time axis
here is the paper's actual OFDM-symbol axis, not an antenna-array proxy.

Three plots (paper Fig. 3 / Fig. 4a / Fig. 4b), same estimator set,
legend, colours and markers as ``gmmce/plotting.py``:
    nmse_vs_snr.png         NMSE vs SNR
    nmse_vs_training.png     NMSE vs training data
    nmse_vs_components.png    NMSE vs GMM components
into ``results_subregion_tf[_comb{s}][_{scale}]/``.

    python run_subregion_tf_experiments.py --scale quick                     # full pilot, smoke test (CPU)
    python run_subregion_tf_experiments.py                                   # full pilot, paper50 (K=128,N=1e5,n_iter=50), GPU (default)
    python run_subregion_tf_experiments.py --comb-spacing 8                  # freq-comb-8 pilot, paper50, GPU
    python run_subregion_tf_experiments.py --no-gpu                          # force CPU
    python run_subregion_tf_experiments.py --replot --comb-spacing 8         # redraw from JSON
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from gmmce.channel_model import SystemConfig
from gmmce.pilots import PilotGrid, selection_matrix
from gmmce.pipeline import (train_all, evaluate_joint, evaluate_cascade, normalized_mse,
                            ALL_GMM_VARIANTS)
from gmmce.pdp_ds import pdp_ds_kron_estimate, pdp_ds_2x1d_estimate
from gmmce.plotting import STYLE, _decade_ticks_only
from gmmce.subregion_tf_data import SubregionChannelsTF

# same component budgets as gmmce/run_experiments.py SCALES / run_subregion_experiments.py
SCALES = {
    "quick":   dict(K=8, Kt=4, Kc=2, Kt_2x1d=2, Kc_2x1d=6, n_train=3000, n_calib=500, n_test=500, n_iter=8),
    "demo":    dict(K=16, Kt=8, Kc=4, Kt_2x1d=4, Kc_2x1d=12, n_train=5000, n_calib=800, n_test=800, n_iter=10),
    "k16n10k": dict(K=16, Kt=4, Kc=4, Kt_2x1d=4, Kc_2x1d=12, n_train=10000, n_calib=1000, n_test=1200, n_iter=12),
    "k16n40k": dict(K=16, Kt=4, Kc=4, Kt_2x1d=4, Kc_2x1d=12, n_train=40000, n_calib=2000, n_test=3000, n_iter=15),
    "paper":   dict(K=128, Kt=8, Kc=16, Kt_2x1d=32, Kc_2x1d=96, n_train=100000, n_calib=5000, n_test=10000, n_iter=25),
    "paper50": dict(K=128, Kt=8, Kc=16, Kt_2x1d=32, Kc_2x1d=96, n_train=100000, n_calib=4000, n_test=5000, n_iter=50),
}

SNR_DB_LIST = [-10, 0, 10, 20, 30]
FIG3_ORDER = ["GMM full", "GMM kron", "GMM b-toep", "GMM b-circ", "GMM Weichselberger",
              "GMM 2x1D", "GMM 2x1D-circ", "GMM 2x1D-toep",
              "PDP+DS 2x1D", "PDP+DS kron"]
FIG4_ORDER = ["GMM full", "GMM kron", "GMM b-toep", "GMM b-circ", "GMM Weichselberger",
              "GMM 2x1D", "GMM 2x1D-toep", "GMM 2x1D-circ"]
FIT_VARIANTS = ALL_GMM_VARIANTS   # overridden to drop "b-toep" by --no-btoep in main()


def _eval_all_joint_and_cascade(models, grid, Hcal, Hte, sigma2, rng, names):
    out = {}
    if "GMM full" in names:
        out["GMM full"], _ = evaluate_joint(models.gmm_full, grid, Hte, sigma2, rng)
    if "GMM b-toep" in names:
        out["GMM b-toep"], _ = evaluate_joint(models.gmm_btoep, grid, Hte, sigma2, rng)
    if "GMM b-circ" in names:
        out["GMM b-circ"], _ = evaluate_joint(models.gmm_bcirc, grid, Hte, sigma2, rng)
    if "GMM kron" in names:
        out["GMM kron"], _ = evaluate_joint(models.gmm_kron, grid, Hte, sigma2, rng)
    if "GMM Weichselberger" in names:
        out["GMM Weichselberger"], _ = evaluate_joint(models.gmm_weichselberger, grid, Hte, sigma2, rng)
    if "GMM 2x1D" in names:
        out["GMM 2x1D"], _ = evaluate_cascade(models.gmm_freq_2x1d, models.gmm_time_2x1d,
                                              grid, Hcal, Hte, sigma2, rng)
    if "GMM 2x1D-toep" in names:
        out["GMM 2x1D-toep"], _ = evaluate_cascade(models.gmm_freq_toep, models.gmm_time_toep,
                                                   grid, Hcal, Hte, sigma2, rng)
    if "GMM 2x1D-circ" in names:
        out["GMM 2x1D-circ"], _ = evaluate_cascade(models.gmm_freq_circ, models.gmm_time_circ,
                                                   grid, Hcal, Hte, sigma2, rng)
    return out


def make_grid(Nc: int, Nt: int, comb_spacing: int = 0) -> PilotGrid:
    """comb_spacing 0 or 1 -> full pilot (A = I).  comb_spacing s>1 -> a
    frequency comb: pilots on every s-th subcarrier, ALL Nt OFDM symbols
    observed (Npc = ceil(Nc/s), Npt = Nt) -- the classic OFDM comb-pilot
    pattern (sparse in frequency, dense in time). Still a separable
    Cartesian-product grid, so the Kronecker / 2x1D estimators' A = A_t (x)
    A_c assumption holds."""
    g = PilotGrid.__new__(PilotGrid)
    g.Nc, g.Nt = Nc, Nt
    s = comb_spacing if comb_spacing and comb_spacing > 1 else 1
    g.pilot_carriers = np.arange(0, Nc, s)
    g.pilot_symbols = np.arange(Nt)
    g.Npc, g.Npt = len(g.pilot_carriers), Nt
    g.A = selection_matrix(Nc, Nt, g.pilot_carriers, g.pilot_symbols)
    g.Np = g.A.shape[0]
    return g


def run_nmse_vs_snr(data, cfg, sc, grid, seed=0, log=print) -> dict:
    Htr, Hcal, Hte = data.train(sc["n_train"]), data.calib(sc["n_calib"]), data.test(sc["n_test"])
    log(f"[snr] train_all K={sc['K']} Kt={sc['Kt']} Kc={sc['Kc']} on {len(Htr)} (D={data.Nc*data.Nt}) ...")
    t0 = time.time()
    models = train_all(Htr, grid, cfg, K=sc["K"], Kt=sc["Kt"], Kc=sc["Kc"],
                       Kt_2x1d=sc["Kt_2x1d"], Kc_2x1d=sc["Kc_2x1d"],
                       n_iter=sc["n_iter"], seed=seed, which=FIT_VARIANTS)
    log(f"[snr] train_all done in {time.time()-t0:.1f}s")

    res = {"snr_db": list(SNR_DB_LIST)}
    names = FIG3_ORDER
    for n in names:
        res[n] = []
    for snr_db in SNR_DB_LIST:
        sigma2 = 10 ** (-snr_db / 10)
        rng = np.random.default_rng(seed + 5000 + int(snr_db))
        joint = _eval_all_joint_and_cascade(models, grid, Hcal, Hte, sigma2, rng, names)
        for n, v in joint.items():
            res[n].append(v)
        H_hat = pdp_ds_kron_estimate(Hte, grid, sigma2, rng)
        res["PDP+DS kron"].append(normalized_mse(H_hat, Hte))
        H_hat, _ = pdp_ds_2x1d_estimate(Hte, grid, sigma2, rng)
        res["PDP+DS 2x1D"].append(normalized_mse(H_hat, Hte))
        log(f"[snr] SNR={snr_db:>4} dB  " + "  ".join(f"{k}={res[k][-1]:.3e}" for k in names))
    return res


def run_nmse_vs_training(data, cfg, sc, grid, snr_db=10.0, seed=0, log=print) -> dict:
    Hcal, Hte = data.calib(sc["n_calib"]), data.test(sc["n_test"])
    sigma2 = 10 ** (-snr_db / 10)
    n_list = sorted(set(int(x) for x in
                        [sc["n_train"] // 20, sc["n_train"] // 8, sc["n_train"] // 3, sc["n_train"]]))
    Htr_max = data.train(max(n_list))
    names = FIG4_ORDER
    out = {"n_train": n_list, **{n: [] for n in names}}
    for n_train in n_list:
        models = train_all(Htr_max[:n_train], grid, cfg, K=sc["K"], Kt=sc["Kt"], Kc=sc["Kc"],
                           Kt_2x1d=sc["Kt_2x1d"], Kc_2x1d=sc["Kc_2x1d"],
                           n_iter=sc["n_iter"], seed=seed, which=FIT_VARIANTS)
        rng = np.random.default_rng(seed + 9000 + n_train)
        joint = _eval_all_joint_and_cascade(models, grid, Hcal, Hte, sigma2, rng, names)
        for n, v in joint.items():
            out[n].append(v)
        log(f"[train] n_train={n_train}  " + "  ".join(f"{k}={out[k][-1]:.3e}" for k in names))
    return out


def run_nmse_vs_components(data, cfg, sc, grid, snr_db=10.0, seed=0, log=print) -> dict:
    Htr, Hcal, Hte = data.train(sc["n_train"]), data.calib(sc["n_calib"]), data.test(sc["n_test"])
    sigma2 = 10 ** (-snr_db / 10)
    K_list = sorted(set(int(x) for x in
                        [max(2, sc["K"] // 16), max(4, sc["K"] // 4), sc["K"] // 2, sc["K"]]))
    names = FIG4_ORDER
    out = {"K": K_list, **{n: [] for n in names}}
    for K in K_list:
        Kt = max(1, int(round(K ** 0.5)))
        Kc = max(1, K // Kt)
        Kt_2x1d = max(1, int(round(K / 4)))
        Kc_2x1d = max(1, K - Kt_2x1d)
        models = train_all(Htr, grid, cfg, K=K, Kt=Kt, Kc=Kc,
                           Kt_2x1d=Kt_2x1d, Kc_2x1d=Kc_2x1d,
                           n_iter=sc["n_iter"], seed=seed, which=FIT_VARIANTS)
        rng = np.random.default_rng(seed + 11000 + K)
        joint = _eval_all_joint_and_cascade(models, grid, Hcal, Hte, sigma2, rng, names)
        for n, v in joint.items():
            out[n].append(v)
        log(f"[comp] K={K} (Kt={Kt},Kc={Kc})  " + "  ".join(f"{k}={out[k][-1]:.3e}" for k in names))
    return out


# ----------------------------------------------------------------------- plots
PLOT_TITLE = ""   # set in main()


def _sci_tick_label(val: float) -> str:
    """'5000' -> '$5 \\times 10^{3}$', '100000' -> '$10^{5}$' -- paper-style
    scientific-notation tick labels for a log axis (matches Fesl Fig. 4a)."""
    if val <= 0:
        return ""
    exp = int(np.floor(np.log10(val) + 1e-9))
    mant = round(val / 10.0 ** exp, 2)
    if abs(mant - round(mant)) < 1e-6:
        mant = int(round(mant))
    if mant == 1:
        return rf"$10^{{{exp}}}$"
    return rf"${mant:g} \times 10^{{{exp}}}$"


def _plot_curves(x, xlabel, results, order, save_path, xticks=None, decade_y=True, sci_xticks=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in order:
        if name in results:
            ax.plot(x, results[name], label=name, **STYLE[name])
    if PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=10)
    ax.set_yscale("log")
    if decade_y:
        _decade_ticks_only(ax, "y")
    else:  # narrow range: explicit 1/2/3/5-per-decade ticks with plain labels
        vals = np.concatenate([results[n] for n in order if n in results])
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        ticks = [m * 10.0 ** e for e in range(-4, 1) for m in (1, 2, 3, 5)]
        ticks = [t for t in ticks if lo / 1.6 <= t <= hi * 1.6]
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    if xticks is not None:
        ax.set_xscale("log")
        ax.set_xticks(xticks)
        if sci_xticks:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _sci_tick_label(v)))
        else:
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlim(min(xticks) * 0.85, max(xticks) * 1.18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized MSE")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {save_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", choices=list(SCALES.keys()), default="paper50",
                   help="default paper50: K=128, N_train=1e5, n_iter=50 (task setting)")
    p.add_argument("--outdir", default=None, help="default: results_subregion_tf")
    p.add_argument("--n-symbols", type=int, default=14, help="Nt: needs train/test_siso_S{s}_..._subregion100.npz")
    p.add_argument("--f", type=int, default=64, help="Nc: subcarriers")
    p.add_argument("--speed", type=int, default=10, help="UE speed (m/s) tag of the data pool to load")
    p.add_argument("--nc-ds", type=int, default=1, help="keep every k-th subcarrier")
    p.add_argument("--nt-ds", type=int, default=1, help="keep every k-th OFDM symbol")
    p.add_argument("--comb-spacing", type=int, default=0,
                   help="0/1 = full pilot; s>1 = frequency comb, pilots every s-th subcarrier "
                        "(all OFDM symbols observed) -> Np = ceil(Nc/s)*Nt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-tag", default="", help="load train/test_siso_S{s}_F{f}_v{v}_subregion100_{tag}.npz "
                   "instead of the plain filename (e.g. 'ulatf' for the ULA-mobility-cube-derived pool)")
    p.add_argument("--gpu", dest="gpu", action="store_true", default=True,
                   help="fit the GMMs on the GPU (torch) -- default ON for this script "
                        "(task setting); ~7x (full) to ~40x (b-toep) faster. Needs the "
                        "sionna-rt/CSI env (torch+CUDA).")
    p.add_argument("--no-gpu", dest="gpu", action="store_false", help="force CPU")
    p.add_argument("--no-btoep", action="store_true",
                   help="skip GMM b-toep entirely (not fit, not plotted/legended) -- it needs "
                        "~10x more EM iterations than every other variant, so this is a large "
                        "time saving when it isn't needed.")
    p.add_argument("--skip", nargs="*", default=[], choices=["fig3", "fig4a", "fig4b"])
    p.add_argument("--replot", action="store_true", help="redraw PNGs from the existing JSONs, no compute")
    args = p.parse_args()

    if args.gpu and not args.replot:
        from gmmce import gpu_gmm
        gpu_gmm.patch()

    global FIG3_ORDER, FIG4_ORDER, FIT_VARIANTS
    if args.no_btoep:
        FIG3_ORDER = [n for n in FIG3_ORDER if n != "GMM b-toep"]
        FIG4_ORDER = [n for n in FIG4_ORDER if n != "GMM b-toep"]
        FIT_VARIANTS = tuple(v for v in ALL_GMM_VARIANTS if v != "b-toep")

    comb = args.comb_spacing if args.comb_spacing and args.comb_spacing > 1 else 0
    default_outdir = (f"results_subregion_tf_v{args.speed}"
                      + (f"_{args.data_tag}" if args.data_tag else "")
                      + (f"_comb{comb}" if comb else "")
                      + ("" if args.scale == "quick" else f"_{args.scale}"))
    args.outdir = args.outdir or default_outdir
    os.makedirs(args.outdir, exist_ok=True)
    sc = SCALES[args.scale]

    global PLOT_TITLE
    _nc = args.f // args.nc_ds
    _nt = args.n_symbols // args.nt_ds
    _npc = len(range(0, _nc, comb)) if comb else _nc
    _sys_label = "1x4 ULA (ant#0)" if args.data_tag == "ulatf" else "SISO"
    PLOT_TITLE = (f"{_sys_label} time-freq, v={args.speed}m/s, Nc={_nc} Nt={_nt}, "
                  + (f"comb spacing {comb}" if comb else "full pilot")
                  + f"  (Np={_npc * _nt}/{_nc * _nt})"
                  + (f"  [{args.scale}: K={sc['K']}, N={sc['n_train']}]"
                     if args.scale not in ("quick", "paper50") else ""))

    if args.replot:
        j = lambda n: json.load(open(os.path.join(args.outdir, n)))
        if "fig3" not in args.skip:
            r = j("nmse_vs_snr.json")
            _plot_curves(r["snr_db"], "SNR [dB]", r, FIG3_ORDER,
                         os.path.join(args.outdir, "nmse_vs_snr.png"))
        if "fig4a" not in args.skip:
            r = j("nmse_vs_training.json")
            _plot_curves(r["n_train"], "Training data", r, FIG4_ORDER,
                         os.path.join(args.outdir, "nmse_vs_training.png"),
                         xticks=r["n_train"], decade_y=True, sci_xticks=True)
        if "fig4b" not in args.skip:
            r = j("nmse_vs_components.json")
            _plot_curves(r["K"], "GMM components", r, FIG4_ORDER,
                         os.path.join(args.outdir, "nmse_vs_components.png"),
                         xticks=r["K"], decade_y=True)
        return

    data = SubregionChannelsTF(n_symbols=args.n_symbols, n_subcarriers=args.f, speed=args.speed,
                               nc_ds=args.nc_ds, nt_ds=args.nt_ds, seed=args.seed, data_tag=args.data_tag)
    cfg = SystemConfig(Nc=data.Nc, Nt=data.Nt)
    grid = make_grid(data.Nc, data.Nt, comb)
    pilot_desc = "full pilot (A=I)" if not comb else f"comb spacing {comb}"
    print(f"subregion100 SISO time-freq channels: Nc(freq)={data.Nc}  Nt(OFDM symbol)={data.Nt}  "
          f"D={data.Nc*data.Nt}  speed={args.speed}m/s  scale={data.scale:.4g}  |  scale={args.scale}  "
          f"gpu={args.gpu}")
    print(f"pilots: {pilot_desc}  ->  Np = {grid.Np} / {data.Nc*data.Nt}  "
          f"(Npc={grid.Npc}, Npt={grid.Npt})")
    t0 = time.time()

    if "fig3" not in args.skip:
        print("=== NMSE vs SNR ===")
        r = run_nmse_vs_snr(data, cfg, sc, grid, seed=args.seed)
        json.dump(r, open(os.path.join(args.outdir, "nmse_vs_snr.json"), "w"), indent=2)
        _plot_curves(r["snr_db"], "SNR [dB]", r, FIG3_ORDER,
                     os.path.join(args.outdir, "nmse_vs_snr.png"))

    if "fig4a" not in args.skip:
        print("=== NMSE vs training data ===")
        r = run_nmse_vs_training(data, cfg, sc, grid, seed=args.seed)
        json.dump(r, open(os.path.join(args.outdir, "nmse_vs_training.json"), "w"), indent=2)
        _plot_curves(r["n_train"], "Training data", r, FIG4_ORDER,
                     os.path.join(args.outdir, "nmse_vs_training.png"),
                     xticks=r["n_train"], decade_y=True, sci_xticks=True)

    if "fig4b" not in args.skip:
        print("=== NMSE vs GMM components ===")
        r = run_nmse_vs_components(data, cfg, sc, grid, seed=args.seed)
        json.dump(r, open(os.path.join(args.outdir, "nmse_vs_components.json"), "w"), indent=2)
        _plot_curves(r["K"], "GMM components", r, FIG4_ORDER,
                     os.path.join(args.outdir, "nmse_vs_components.png"),
                     xticks=r["K"], decade_y=True)

    print(f"Done in {time.time()-t0:.1f}s. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
