"""
Additive-only companion to ``run_subregion_experiments.py`` /
``run_subregion_tf_pilot_experiments.py``: fits ONLY the new
"GMM Weichselberger" structured-covariance estimator (see
``Share/exp2_guideline.md``) on top of the already-completed 1x4 ULA
velocity-sweep results, and merges its NMSE curve into the existing
JSON/PNG outputs. Every other estimator (full/kron/b-circ/2x1D/...) is
read from the existing JSON, never recomputed -- per the guideline's
"do not re-run other structures that have already been completed"
condition. Same data/scale/seed as those completed runs (paper50, GPU by
default, ``--data-tag ulatf``).

Four targets:
    --target m4comb8                       -> results_subregion_M4_ulatf_comb8_paper50/
    --target m4fullpilot                   -> results_subregion_M4_ulatf_paper50/
    --target tfpilot --speed 0 3 6 10      -> results_subregion_tf_v{speed}_ulatf_{circ,unif}_c8t2_paper50/
        (one Weichselberger GMM fit per velocity, shared across both the
        circ_c8t2 and unif_c8t2 output dirs at each x-value -- the GMM
        fit never sees the pilot pattern, exactly like
        run_subregion_tf_pilot_experiments.py's original run does.)
    --target tffullpilot --speed 0 3 6 10  -> results_subregion_tf_v{speed}_ulatf_paper50/
        (full pilot, A=I -- single directory per velocity, no unif/circ
        split, using run_subregion_tf_experiments.py's own grid/plotting.)

    python run_weichselberger_addon.py --target m4comb8
    python run_weichselberger_addon.py --target m4fullpilot
    python run_weichselberger_addon.py --target tfpilot --speed 0 3 6 10
    python run_weichselberger_addon.py --target tffullpilot --speed 0 3 6 10
    python run_weichselberger_addon.py --target tfpilot --speed 0 3 6 10 --no-gpu
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np

from gmmce.channel_model import SystemConfig
from gmmce.pipeline import train_all, evaluate_joint
from gmmce.subregion_data import SubregionChannels
from gmmce.subregion_tf_data import SubregionChannelsTF

import run_subregion_experiments as rse
import run_subregion_tf_experiments as rt
import run_subregion_tf_pilot_experiments as rtp

NAME = "GMM Weichselberger"
# Offset so this addon's noise realizations are distinct from (but just as
# valid as) the ones the original run drew -- each estimator in this
# codebase already gets its own independent noise draw at a given SNR
# (see gmmce/pipeline.py's evaluate_joint call sites), so this is
# consistent with, not a deviation from, the existing methodology.
NOISE_OFFSET = 90000


def _fit(Htr, grid, cfg, sc, seed, K=None):
    K = sc["K"] if K is None else K
    Kt = sc["Kt"] if K == sc["K"] else max(1, int(round(K ** 0.5)))
    Kc = sc["Kc"] if K == sc["K"] else max(1, K // Kt)
    Kt_2x1d = sc["Kt_2x1d"] if K == sc["K"] else max(1, int(round(K / 4)))
    Kc_2x1d = sc["Kc_2x1d"] if K == sc["K"] else max(1, K - Kt_2x1d)
    m = train_all(Htr, grid, cfg, K=K, Kt=Kt, Kc=Kc, Kt_2x1d=Kt_2x1d, Kc_2x1d=Kc_2x1d,
                  n_iter=sc["n_iter"], seed=seed, which=("Weichselberger",))
    return m.gmm_weichselberger


def _load(path):
    return json.load(open(path))


def _save(path, res):
    json.dump(res, open(path, "w"), indent=2)


# --------------------------------------------------------------------- m4 (space-freq)

def _m4_title(comb, nc_ds=1, m=4):
    _nc = 64 // nc_ds
    pilot = f"freq comb spacing {comb}" if comb else "full pilot"
    _npc = len(range(0, _nc, comb)) if comb else _nc
    return f"1x{m} TX, {pilot}  (Np={_npc * m}/{_nc * m})"


def run_m4(outdir, comb_spacing, tag, seed=0, log=print):
    sc = rse.SCALES["paper50"]
    data = SubregionChannels(m=4, nc_ds=1, nt_ds=1, seed=seed, data_tag="ulatf")
    cfg = SystemConfig(Nc=data.Nc, Nt=data.Nt)
    grid = rse.make_grid(data.Nc, data.Nt, comb_spacing=comb_spacing)

    snr_path = os.path.join(outdir, "nmse_vs_snr.json")
    res_snr = _load(snr_path)
    if NAME not in res_snr:
        Htr, Hte = data.train(sc["n_train"]), data.test(sc["n_test"])
        log(f"[{tag}][snr] fitting {NAME} on {len(Htr)} samples (D={data.Nc*data.Nt}) ...")
        t0 = time.time()
        gmm = _fit(Htr, grid, cfg, sc, seed)
        log(f"[{tag}][snr] fit done in {time.time()-t0:.1f}s")
        vals = []
        for snr_db in res_snr["snr_db"]:
            sigma2 = 10 ** (-snr_db / 10)
            rng = np.random.default_rng(seed + 5000 + int(snr_db) + NOISE_OFFSET)
            mse, _ = evaluate_joint(gmm, grid, Hte, sigma2, rng)
            vals.append(mse)
            log(f"[{tag}][snr] SNR={snr_db:>4} dB  {NAME}={mse:.3e}")
        res_snr[NAME] = vals
        _save(snr_path, res_snr)
    else:
        log(f"[{tag}][snr] {NAME} already present, skipping")

    train_path = os.path.join(outdir, "nmse_vs_training.json")
    res_train = _load(train_path)
    if NAME not in res_train:
        Hte = data.test(sc["n_test"])
        Htr_max = data.train(max(res_train["n_train"]))
        vals = []
        for n_train in res_train["n_train"]:
            gmm = _fit(Htr_max[:n_train], grid, cfg, sc, seed)
            rng = np.random.default_rng(seed + 9000 + n_train + NOISE_OFFSET)
            mse, _ = evaluate_joint(gmm, grid, Hte, 10 ** (-10 / 10), rng)
            vals.append(mse)
            log(f"[{tag}][train] n_train={n_train}  {NAME}={mse:.3e}")
        res_train[NAME] = vals
        _save(train_path, res_train)
    else:
        log(f"[{tag}][train] {NAME} already present, skipping")

    comp_path = os.path.join(outdir, "nmse_vs_components.json")
    res_comp = _load(comp_path)
    if NAME not in res_comp:
        Htr, Hte = data.train(sc["n_train"]), data.test(sc["n_test"])
        vals = []
        for K in res_comp["K"]:
            gmm = _fit(Htr, grid, cfg, sc, seed, K=K)
            rng = np.random.default_rng(seed + 11000 + K + NOISE_OFFSET)
            mse, _ = evaluate_joint(gmm, grid, Hte, 10 ** (-10 / 10), rng)
            vals.append(mse)
            log(f"[{tag}][comp] K={K}  {NAME}={mse:.3e}")
        res_comp[NAME] = vals
        _save(comp_path, res_comp)
    else:
        log(f"[{tag}][comp] {NAME} already present, skipping")

    rse.PLOT_TITLE = _m4_title(comb_spacing)
    r = _load(snr_path)
    rse._plot_curves(r["snr_db"], "SNR [dB]", r, rse.FIG3_ORDER, os.path.join(outdir, "nmse_vs_snr.png"))
    r = _load(train_path)
    rse._plot_curves(r["n_train"], "Training data", r, rse.FIG4_ORDER, os.path.join(outdir, "nmse_vs_training.png"),
                      xticks=r["n_train"], decade_y=True, sci_xticks=True)
    r = _load(comp_path)
    rse._plot_curves(r["K"], "GMM components", r, rse.FIG4_ORDER, os.path.join(outdir, "nmse_vs_components.png"),
                      xticks=r["K"], decade_y=True)
    log(f"[{tag}] done -> {outdir}/")


def run_m4comb8(seed=0, log=print):
    run_m4("results_subregion_M4_ulatf_comb8_paper50", 8, "m4comb8", seed=seed, log=log)


def run_m4_fullpilot(seed=0, log=print):
    run_m4("results_subregion_M4_ulatf_paper50", 0, "m4fullpilot", seed=seed, log=log)


# --------------------------------------------------------------------- tf full pilot (single dir, no unif/circ split)

def _tf_fullpilot_title(speed, Nc, Nt):
    return (f"1x4 ULA (ant#0) time-freq, v={speed}m/s, Nc={Nc} Nt={Nt}, "
            f"full pilot  (Np={Nc * Nt}/{Nc * Nt})")


def run_tf_fullpilot(speed, seed=0, log=print):
    outdir = f"results_subregion_tf_v{speed}_ulatf_paper50"
    tag = f"tf v{speed} fullpilot"
    sc = rt.SCALES["paper50"]
    data = SubregionChannelsTF(n_symbols=14, n_subcarriers=64, speed=speed, seed=seed, data_tag="ulatf")
    cfg = SystemConfig(Nc=data.Nc, Nt=data.Nt)
    grid = rt.make_grid(data.Nc, data.Nt, comb_spacing=0)

    snr_path = os.path.join(outdir, "nmse_vs_snr.json")
    res_snr = _load(snr_path)
    if NAME not in res_snr:
        Htr, Hte = data.train(sc["n_train"]), data.test(sc["n_test"])
        log(f"[{tag}][snr] fitting {NAME} on {len(Htr)} samples (D={data.Nc*data.Nt}) ...")
        t0 = time.time()
        gmm = _fit(Htr, grid, cfg, sc, seed)
        log(f"[{tag}][snr] fit done in {time.time()-t0:.1f}s")
        vals = []
        for snr_db in res_snr["snr_db"]:
            sigma2 = 10 ** (-snr_db / 10)
            rng = np.random.default_rng(seed + 5000 + int(snr_db) + NOISE_OFFSET)
            mse, _ = evaluate_joint(gmm, grid, Hte, sigma2, rng)
            vals.append(mse)
            log(f"[{tag}][snr] SNR={snr_db:>4} dB  {NAME}={mse:.3e}")
        res_snr[NAME] = vals
        _save(snr_path, res_snr)
    else:
        log(f"[{tag}][snr] {NAME} already present, skipping")

    train_path = os.path.join(outdir, "nmse_vs_training.json")
    res_train = _load(train_path)
    if NAME not in res_train:
        Hte = data.test(sc["n_test"])
        Htr_max = data.train(max(res_train["n_train"]))
        vals = []
        for n_train in res_train["n_train"]:
            gmm = _fit(Htr_max[:n_train], grid, cfg, sc, seed)
            rng = np.random.default_rng(seed + 9000 + n_train + NOISE_OFFSET)
            mse, _ = evaluate_joint(gmm, grid, Hte, 10 ** (-10 / 10), rng)
            vals.append(mse)
            log(f"[{tag}][train] n_train={n_train}  {NAME}={mse:.3e}")
        res_train[NAME] = vals
        _save(train_path, res_train)
    else:
        log(f"[{tag}][train] {NAME} already present, skipping")

    comp_path = os.path.join(outdir, "nmse_vs_components.json")
    res_comp = _load(comp_path)
    if NAME not in res_comp:
        Htr, Hte = data.train(sc["n_train"]), data.test(sc["n_test"])
        vals = []
        for K in res_comp["K"]:
            gmm = _fit(Htr, grid, cfg, sc, seed, K=K)
            rng = np.random.default_rng(seed + 11000 + K + NOISE_OFFSET)
            mse, _ = evaluate_joint(gmm, grid, Hte, 10 ** (-10 / 10), rng)
            vals.append(mse)
            log(f"[{tag}][comp] K={K}  {NAME}={mse:.3e}")
        res_comp[NAME] = vals
        _save(comp_path, res_comp)
    else:
        log(f"[{tag}][comp] {NAME} already present, skipping")

    rt.PLOT_TITLE = _tf_fullpilot_title(speed, data.Nc, data.Nt)
    r = _load(snr_path)
    rt._plot_curves(r["snr_db"], "SNR [dB]", r, rt.FIG3_ORDER, os.path.join(outdir, "nmse_vs_snr.png"))
    r = _load(train_path)
    rt._plot_curves(r["n_train"], "Training data", r, rt.FIG4_ORDER, os.path.join(outdir, "nmse_vs_training.png"),
                     xticks=r["n_train"], decade_y=True, sci_xticks=True)
    r = _load(comp_path)
    rt._plot_curves(r["K"], "GMM components", r, rt.FIG4_ORDER, os.path.join(outdir, "nmse_vs_components.png"),
                     xticks=r["K"], decade_y=True)
    log(f"[{tag}] done -> {outdir}/")


# --------------------------------------------------------------------- tfpilot

def run_tfpilot(speed, seed=0, log=print):
    Nc, Nt = 64, 14
    sc = rt.SCALES["paper50"]
    args_ns = SimpleNamespace(scale="paper50", speed=speed, patterns=["uniform", "circular"],
                               carrier_spacing=8, time_spacing=2, shift=1)
    pats = rtp.build_patterns(args_ns, Nc, Nt)
    data = SubregionChannelsTF(n_symbols=Nt, n_subcarriers=Nc, speed=speed, seed=seed, data_tag="ulatf")
    cfg = SystemConfig(Nc=data.Nc, Nt=data.Nt)

    # ---- nmse_vs_snr : one fit shared by both patterns ----
    snr_res = {p["tag"]: _load(os.path.join(p["outdir"], "nmse_vs_snr.json")) for p in pats}
    if any(NAME not in snr_res[p["tag"]] for p in pats):
        Htr, Hte = data.train(sc["n_train"]), data.test(sc["n_test"])
        log(f"[tf v{speed}][snr] fitting {NAME} once on {len(Htr)} samples (D={data.Nc*data.Nt}) ...")
        t0 = time.time()
        gmm = _fit(Htr, pats[0]["grid"], cfg, sc, seed)
        log(f"[tf v{speed}][snr] fit done in {time.time()-t0:.1f}s")
        for p in pats:
            res = snr_res[p["tag"]]
            if NAME in res:
                continue
            vals = []
            for snr_db in res["snr_db"]:
                sigma2 = 10 ** (-snr_db / 10)
                rng = np.random.default_rng(seed + 5000 + int(snr_db) + NOISE_OFFSET)
                mse, _ = evaluate_joint(gmm, p["grid"], Hte, sigma2, rng)
                vals.append(mse)
                log(f"[tf v{speed}][snr] {p['tag']} SNR={snr_db:>4} dB  {NAME}={mse:.3e}")
            res[NAME] = vals
            _save(os.path.join(p["outdir"], "nmse_vs_snr.json"), res)
    else:
        log(f"[tf v{speed}][snr] {NAME} already present in both dirs, skipping")

    # ---- nmse_vs_training : one fit per n_train, shared by both patterns ----
    train_res = {p["tag"]: _load(os.path.join(p["outdir"], "nmse_vs_training.json")) for p in pats}
    if any(NAME not in train_res[p["tag"]] for p in pats):
        Hte = data.test(sc["n_test"])
        n_list = train_res[pats[0]["tag"]]["n_train"]
        Htr_max = data.train(max(n_list))
        for n_train in n_list:
            gmm = _fit(Htr_max[:n_train], pats[0]["grid"], cfg, sc, seed)
            for p in pats:
                res = train_res[p["tag"]]
                if NAME in res:
                    continue
                res.setdefault(f"_{NAME}_tmp", []).append(None)
                rng = np.random.default_rng(seed + 9000 + n_train + NOISE_OFFSET)
                mse, _ = evaluate_joint(gmm, p["grid"], Hte, 10 ** (-10 / 10), rng)
                res[f"_{NAME}_tmp"][-1] = mse
                log(f"[tf v{speed}][train] {p['tag']} n_train={n_train}  {NAME}={mse:.3e}")
        for p in pats:
            res = train_res[p["tag"]]
            if f"_{NAME}_tmp" in res:
                res[NAME] = res.pop(f"_{NAME}_tmp")
                _save(os.path.join(p["outdir"], "nmse_vs_training.json"), res)
    else:
        log(f"[tf v{speed}][train] {NAME} already present in both dirs, skipping")

    # ---- nmse_vs_components : one fit per K, shared by both patterns ----
    comp_res = {p["tag"]: _load(os.path.join(p["outdir"], "nmse_vs_components.json")) for p in pats}
    if any(NAME not in comp_res[p["tag"]] for p in pats):
        Htr, Hte = data.train(sc["n_train"]), data.test(sc["n_test"])
        K_list = comp_res[pats[0]["tag"]]["K"]
        for K in K_list:
            gmm = _fit(Htr, pats[0]["grid"], cfg, sc, seed, K=K)
            for p in pats:
                res = comp_res[p["tag"]]
                if NAME in res:
                    continue
                res.setdefault(f"_{NAME}_tmp", []).append(None)
                rng = np.random.default_rng(seed + 11000 + K + NOISE_OFFSET)
                mse, _ = evaluate_joint(gmm, p["grid"], Hte, 10 ** (-10 / 10), rng)
                res[f"_{NAME}_tmp"][-1] = mse
                log(f"[tf v{speed}][comp] {p['tag']} K={K}  {NAME}={mse:.3e}")
        for p in pats:
            res = comp_res[p["tag"]]
            if f"_{NAME}_tmp" in res:
                res[NAME] = res.pop(f"_{NAME}_tmp")
                _save(os.path.join(p["outdir"], "nmse_vs_components.json"), res)
    else:
        log(f"[tf v{speed}][comp] {NAME} already present in both dirs, skipping")

    for p in pats:
        ttl = rtp._title(args_ns, p, Nc, Nt, sc)
        rt.PLOT_TITLE = ttl
        r = _load(os.path.join(p["outdir"], "nmse_vs_snr.json"))
        rt._plot_curves(r["snr_db"], "SNR [dB]", r, p["fig3"], os.path.join(p["outdir"], "nmse_vs_snr.png"),
                         decade_y=rtp._decade(r, p["fig3"]))
        r = _load(os.path.join(p["outdir"], "nmse_vs_training.json"))
        rt._plot_curves(r["n_train"], "Training data", r, p["fig4"],
                         os.path.join(p["outdir"], "nmse_vs_training.png"),
                         xticks=r["n_train"], decade_y=rtp._decade(r, p["fig4"]), sci_xticks=True)
        r = _load(os.path.join(p["outdir"], "nmse_vs_components.json"))
        rt._plot_curves(r["K"], "GMM components", r, p["fig4"],
                         os.path.join(p["outdir"], "nmse_vs_components.png"),
                         xticks=r["K"], decade_y=rtp._decade(r, p["fig4"]))
        log(f"[tf v{speed}] done -> {p['outdir']}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True,
                     choices=["m4comb8", "m4fullpilot", "tfpilot", "tffullpilot"])
    ap.add_argument("--speed", type=int, nargs="+", default=[0, 3, 6, 10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", dest="gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="gpu", action="store_false")
    args = ap.parse_args()

    if args.gpu:
        from gmmce import gpu_gmm
        gpu_gmm.patch()

    t0 = time.time()
    if args.target == "m4comb8":
        run_m4comb8(seed=args.seed)
    elif args.target == "m4fullpilot":
        run_m4_fullpilot(seed=args.seed)
    elif args.target == "tfpilot":
        for v in args.speed:
            run_tfpilot(v, seed=args.seed)
    else:
        for v in args.speed:
            run_tf_fullpilot(v, seed=args.seed)
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
