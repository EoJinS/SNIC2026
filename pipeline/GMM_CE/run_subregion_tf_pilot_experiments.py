"""
Pilot-PATTERN experiment on the 1x4 ULA time-freq slice (``--data-tag ulatf``,
antenna-0 slice, Nc=64 subcarriers x Nt=14 OFDM symbols, D=896): same data /
scale / estimators / plots as ``run_subregion_tf_experiments.py``, but the
pilot lattice is sparse on BOTH axes and comes in two flavours. The GMMs are
fit ONCE per velocity (fitting never sees the pilot pattern) and every
pattern is evaluated on the same fitted models.

  uniform  ("unif_c8t2"): pilots on carriers 0,8,...,56 (spacing 8) x OFDM
           symbols 0,2,...,12 (spacing 2) -> a separable Cartesian grid,
           Np = 8*7 = 56 / 896. Full estimator set (b-toep excluded): GMM full / kron /
           b-circ / 2x1D / 2x1D-toep / 2x1D-circ + PDP+DS kron / 2x1D.
  circular ("circ_c8t2"): same 7 pilot symbols 0,2,...,12, but the k-th pilot
           symbol uses carriers (k*shift + 8m) mod 8-comb, i.e. the comb is
           cyclically shifted by ``--shift`` (default 1) carrier per pilot
           symbol (offsets 0..6 -> 56/64 carriers ever observed; offset 7
           never). NOT separable (carrier set differs per pilot symbol), so
           only the joint estimators that work with an arbitrary selection
           matrix A run: GMM full / kron / b-circ. The 2x1D family and the
           PDP+DS baselines assume one shared carrier set and are skipped.

Outputs (per velocity, per pattern):
    results_subregion_tf_v{v}_ulatf_{unif_c8t2|circ_c8t2}_{scale}/
        nmse_vs_snr | nmse_vs_training | nmse_vs_components  (.png/.json)

    python run_subregion_tf_pilot_experiments.py --speed 10                # paper50, GPU
    python run_subregion_tf_pilot_experiments.py --speed 10 --replot       # redraw from JSON
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import run_subregion_tf_experiments as rt
from gmmce.channel_model import SystemConfig
from gmmce.pilots import PilotGrid
from gmmce.pipeline import train_all, normalized_mse, ALL_GMM_VARIANTS
from gmmce.pdp_ds import pdp_ds_kron_estimate, pdp_ds_2x1d_estimate
from gmmce.subregion_tf_data import SubregionChannelsTF

JOINT_ONLY = ("GMM full", "GMM kron", "GMM b-circ", "GMM Weichselberger")


def make_pilot_grid(Nc, Nt, carrier_spacing, time_spacing, circular, shift=1) -> PilotGrid:
    g = PilotGrid.__new__(PilotGrid)
    g.Nc, g.Nt = Nc, Nt
    g.pilot_symbols = np.arange(0, Nt, time_spacing)
    base = np.arange(0, Nc, carrier_spacing)
    g.Npc, g.Npt = len(base), len(g.pilot_symbols)
    rows = []
    for k, t in enumerate(g.pilot_symbols):
        off = (k * shift) % carrier_spacing if circular else 0
        carriers = base + off
        assert carriers.max() < Nc
        rows.append((t, carriers))
    g.Np = g.Npc * g.Npt
    g.A = np.zeros((g.Np, Nc * Nt))
    r = 0
    for t, carriers in rows:                     # symbol-major, same row order as selection_matrix
        for c in carriers:
            g.A[r, t * Nc + c] = 1.0
            r += 1
    if circular:
        g.pilot_carriers = None                  # differs per pilot symbol
        g.pilot_carriers_per_symbol = {int(t): c for t, c in rows}
    else:
        g.pilot_carriers = base
    return g


def build_patterns(args, Nc, Nt):
    pats = []
    for kind in args.patterns:
        circ = kind == "circular"
        grid = make_pilot_grid(Nc, Nt, args.carrier_spacing, args.time_spacing, circ, args.shift)
        tag = f"{'circ' if circ else 'unif'}_c{args.carrier_spacing}t{args.time_spacing}"
        fig3 = [n for n in rt.FIG3_ORDER if n != "GMM b-toep" and (not circ or n in JOINT_ONLY)]
        fig4 = [n for n in rt.FIG4_ORDER if n != "GMM b-toep" and (not circ or n in JOINT_ONLY)]
        desc = (f"circular comb (carrier {args.carrier_spacing}, +{args.shift}/pilot symbol; "
                f"symbol {args.time_spacing})" if circ else
                f"uniform comb (carrier {args.carrier_spacing}, symbol {args.time_spacing})")
        pats.append(dict(kind=kind, tag=tag, grid=grid, fig3=fig3, fig4=fig4, genie=not circ, desc=desc,
                         outdir=f"results_subregion_tf_v{args.speed}_ulatf_{tag}_{args.scale}"))
    return pats


def _title(args, pat, Nc, Nt, sc):
    t = (f"1x4 ULA (ant#0) time-freq, v={args.speed}m/s, Nc={Nc} Nt={Nt}\n{pat['desc']}"
         f"  (Np={pat['grid'].Np}/{Nc * Nt})")
    if args.scale not in ("quick", "paper50"):
        t += f"  [{args.scale}: K={sc['K']}, N={sc['n_train']}]"
    return t


def _decade(res, order):
    vals = np.concatenate([np.asarray(res[n], float) for n in order if n in res])
    return bool(np.nanmax(vals) / np.nanmin(vals) >= 10)


def _dump(pat, name, res, xkey, xlabel, order, title, **plot_kw):
    os.makedirs(pat["outdir"], exist_ok=True)
    json.dump(res, open(os.path.join(pat["outdir"], f"{name}.json"), "w"), indent=2)
    rt.PLOT_TITLE = title
    plot_kw["decade_y"] = _decade(res, order)
    rt._plot_curves(res[xkey], xlabel, res, order, os.path.join(pat["outdir"], f"{name}.png"), **plot_kw)


def run_snr(data, cfg, sc, pats, fit, seed, log=print):
    Htr, Hcal, Hte = data.train(sc["n_train"]), data.calib(sc["n_calib"]), data.test(sc["n_test"])
    log(f"[snr] train_all ONCE for {len(pats)} pilot patterns: K={sc['K']} on {len(Htr)} (D={data.Nc * data.Nt}) ...")
    t0 = time.time()
    models = train_all(Htr, pats[0]["grid"], cfg, K=sc["K"], Kt=sc["Kt"], Kc=sc["Kc"],
                       Kt_2x1d=sc["Kt_2x1d"], Kc_2x1d=sc["Kc_2x1d"],
                       n_iter=sc["n_iter"], seed=seed, which=fit)
    log(f"[snr] train_all done in {time.time() - t0:.1f}s")
    res = {p["tag"]: {"snr_db": list(rt.SNR_DB_LIST), **{n: [] for n in p["fig3"]}} for p in pats}
    for snr_db in rt.SNR_DB_LIST:
        sigma2 = 10 ** (-snr_db / 10)
        for p in pats:
            rng = np.random.default_rng(seed + 5000 + int(snr_db))
            r = res[p["tag"]]
            joint = rt._eval_all_joint_and_cascade(models, p["grid"], Hcal, Hte, sigma2, rng, p["fig3"])
            for n, v in joint.items():
                r[n].append(v)
            if p["genie"]:
                r["PDP+DS kron"].append(normalized_mse(pdp_ds_kron_estimate(Hte, p["grid"], sigma2, rng), Hte))
                H_hat, _ = pdp_ds_2x1d_estimate(Hte, p["grid"], sigma2, rng)
                r["PDP+DS 2x1D"].append(normalized_mse(H_hat, Hte))
            log(f"[snr] {p['tag']} SNR={snr_db:>4} dB  " + "  ".join(f"{k}={r[k][-1]:.3e}" for k in p["fig3"]))
    return res


def run_training(data, cfg, sc, pats, fit, seed, snr_db=10.0, log=print):
    Hcal, Hte = data.calib(sc["n_calib"]), data.test(sc["n_test"])
    sigma2 = 10 ** (-snr_db / 10)
    n_list = sorted(set(int(x) for x in [sc["n_train"] // 20, sc["n_train"] // 8, sc["n_train"] // 3, sc["n_train"]]))
    Htr_max = data.train(max(n_list))
    res = {p["tag"]: {"n_train": n_list, **{n: [] for n in p["fig4"]}} for p in pats}
    for n_train in n_list:
        models = train_all(Htr_max[:n_train], pats[0]["grid"], cfg, K=sc["K"], Kt=sc["Kt"], Kc=sc["Kc"],
                           Kt_2x1d=sc["Kt_2x1d"], Kc_2x1d=sc["Kc_2x1d"],
                           n_iter=sc["n_iter"], seed=seed, which=fit)
        for p in pats:
            rng = np.random.default_rng(seed + 9000 + n_train)
            joint = rt._eval_all_joint_and_cascade(models, p["grid"], Hcal, Hte, sigma2, rng, p["fig4"])
            for n, v in joint.items():
                res[p["tag"]][n].append(v)
            log(f"[train] {p['tag']} n_train={n_train}  "
                + "  ".join(f"{k}={res[p['tag']][k][-1]:.3e}" for k in p["fig4"]))
    return res


def run_components(data, cfg, sc, pats, fit, seed, snr_db=10.0, log=print):
    Htr, Hcal, Hte = data.train(sc["n_train"]), data.calib(sc["n_calib"]), data.test(sc["n_test"])
    sigma2 = 10 ** (-snr_db / 10)
    K_list = sorted(set(int(x) for x in [max(2, sc["K"] // 16), max(4, sc["K"] // 4), sc["K"] // 2, sc["K"]]))
    res = {p["tag"]: {"K": K_list, **{n: [] for n in p["fig4"]}} for p in pats}
    for K in K_list:
        Kt = max(1, int(round(K ** 0.5)))
        Kc = max(1, K // Kt)
        Kt_2x1d = max(1, int(round(K / 4)))
        Kc_2x1d = max(1, K - Kt_2x1d)
        models = train_all(Htr, pats[0]["grid"], cfg, K=K, Kt=Kt, Kc=Kc, Kt_2x1d=Kt_2x1d, Kc_2x1d=Kc_2x1d,
                           n_iter=sc["n_iter"], seed=seed, which=fit)
        for p in pats:
            rng = np.random.default_rng(seed + 11000 + K)
            joint = rt._eval_all_joint_and_cascade(models, p["grid"], Hcal, Hte, sigma2, rng, p["fig4"])
            for n, v in joint.items():
                res[p["tag"]][n].append(v)
            log(f"[comp] {p['tag']} K={K}  " + "  ".join(f"{k}={res[p['tag']][k][-1]:.3e}" for k in p["fig4"]))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", choices=list(rt.SCALES.keys()), default="paper50")
    p.add_argument("--speed", type=int, required=True, help="UE speed (m/s) tag of the ulatf data pool")
    p.add_argument("--patterns", nargs="+", choices=["uniform", "circular"], default=["uniform", "circular"])
    p.add_argument("--carrier-spacing", type=int, default=8)
    p.add_argument("--time-spacing", type=int, default=2)
    p.add_argument("--shift", type=int, default=1, help="circular: carrier shift per pilot symbol")
    p.add_argument("--data-tag", default="ulatf")
    p.add_argument("--n-symbols", type=int, default=14)
    p.add_argument("--f", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", dest="gpu", action="store_true", default=True)
    p.add_argument("--no-gpu", dest="gpu", action="store_false")
    p.add_argument("--skip", nargs="*", default=[], choices=["fig3", "fig4a", "fig4b"])
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()
    sc = rt.SCALES[args.scale]
    Nc, Nt = args.f, args.n_symbols
    pats = build_patterns(args, Nc, Nt)

    if args.replot:
        for pat in pats:
            j = lambda n: json.load(open(os.path.join(pat["outdir"], n)))
            ttl = _title(args, pat, Nc, Nt, sc)
            rt.PLOT_TITLE = ttl
            if "fig3" not in args.skip:
                r = j("nmse_vs_snr.json")
                rt._plot_curves(r["snr_db"], "SNR [dB]", r, pat["fig3"], os.path.join(pat["outdir"], "nmse_vs_snr.png"),
                                decade_y=_decade(r, pat["fig3"]))
            if "fig4a" not in args.skip:
                r = j("nmse_vs_training.json")
                rt._plot_curves(r["n_train"], "Training data", r, pat["fig4"],
                                os.path.join(pat["outdir"], "nmse_vs_training.png"),
                                xticks=r["n_train"], decade_y=_decade(r, pat["fig4"]), sci_xticks=True)
            if "fig4b" not in args.skip:
                r = j("nmse_vs_components.json")
                rt._plot_curves(r["K"], "GMM components", r, pat["fig4"],
                                os.path.join(pat["outdir"], "nmse_vs_components.png"),
                                xticks=r["K"], decade_y=_decade(r, pat["fig4"]))
        return

    if args.gpu:
        from gmmce import gpu_gmm
        gpu_gmm.patch()

    needed = {n for pat in pats for n in pat["fig3"] + pat["fig4"] if n.startswith("GMM ")}
    fit = tuple(v for v in ALL_GMM_VARIANTS if f"GMM {v}" in needed)

    data = SubregionChannelsTF(n_symbols=Nt, n_subcarriers=Nc, speed=args.speed, seed=args.seed,
                               data_tag=args.data_tag)
    cfg = SystemConfig(Nc=data.Nc, Nt=data.Nt)
    print(f"ulatf time-freq  v={args.speed}m/s  Nc={data.Nc} Nt={data.Nt} D={data.Nc * data.Nt}  "
          f"scale={args.scale}  gpu={args.gpu}  fit variants={fit}")
    for pat in pats:
        g = pat["grid"]
        print(f"pattern {pat['tag']}: Np={g.Np}/{data.Nc * data.Nt} (Npc={g.Npc}, Npt={g.Npt}) "
              f"pilot symbols={g.pilot_symbols.tolist()}  -> {pat['outdir']}")
        if pat["kind"] == "circular":
            print("   carriers per pilot symbol:",
                  {t: c.tolist() for t, c in g.pilot_carriers_per_symbol.items()})
        else:
            print("   pilot carriers:", g.pilot_carriers.tolist())
    t0 = time.time()

    if "fig3" not in args.skip:
        print("=== NMSE vs SNR ===")
        res = run_snr(data, cfg, sc, pats, fit, args.seed)
        for pat in pats:
            _dump(pat, "nmse_vs_snr", res[pat["tag"]], "snr_db", "SNR [dB]", pat["fig3"],
                  _title(args, pat, data.Nc, data.Nt, sc))
    if "fig4a" not in args.skip:
        print("=== NMSE vs training data ===")
        res = run_training(data, cfg, sc, pats, fit, args.seed)
        for pat in pats:
            r = res[pat["tag"]]
            _dump(pat, "nmse_vs_training", r, "n_train", "Training data", pat["fig4"],
                  _title(args, pat, data.Nc, data.Nt, sc), xticks=r["n_train"], decade_y=True, sci_xticks=True)
    if "fig4b" not in args.skip:
        print("=== NMSE vs GMM components ===")
        res = run_components(data, cfg, sc, pats, fit, args.seed)
        for pat in pats:
            r = res[pat["tag"]]
            _dump(pat, "nmse_vs_components", r, "K", "GMM components", pat["fig4"],
                  _title(args, pat, data.Nc, data.Nt, sc), xticks=r["K"], decade_y=True)
    print(f"Done in {time.time() - t0:.1f}s. Results in " + ", ".join(p["outdir"] for p in pats) + "/")


if __name__ == "__main__":
    main()
