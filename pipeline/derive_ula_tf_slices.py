"""
Slice the 1xM TX ULA mobility cubes (``generate_channels_subregion100_ula_tf.py``,
h[p].reshape(M, S, F), antenna-major/symbol-mid/subcarrier-minor) into the
two 2-D pools ``run_subregion_experiments.py`` / ``run_subregion_tf_experiments.py``
actually consume -- see ``guideline.md``'s 2026-09-16 note and that script's
docstring for why this is one system sliced two ways, not two systems:

  - space-freq (antenna x freq): the OFDM-symbol-0 slice of the ``--speed 0``
    cube -> ``train/test_M{m}_F64_subregion100_ulatf.npz`` (same layout
    ``gmmce/subregion_data.py`` already expects: (N, M*F), antenna-major).
  - time-freq (freq x OFDM symbol): one fixed TX antenna's slice of each
    ``--speed v`` cube -> ``train/test_siso_S{s}_F{f}_v{v}_subregion100_ulatf.npz``
    (same layout ``gmmce/subregion_tf_data.py`` already expects: (N, S*F),
    OFDM-symbol-major).

Both write with the ``_ulatf`` ``--data-tag`` so they sit alongside (not
overwrite) the pre-existing ``train/test_M4_F64_subregion100.npz`` and
``train/test_siso_S14_F64_v10_subregion100.npz`` pools from the older,
separate-system scripts.

    python3 derive_ula_tf_slices.py --m 4 --speeds 0 3 6 10 --antenna 0
"""
import argparse
import os

import numpy as np

OUT_DIR = "../Data"


def _load(tag, m, s, f, speed):
    path = f"{OUT_DIR}/{tag}_ula{m}_S{s}_F{f}_v{speed}_subregion100.npz"
    d = np.load(path)
    h = np.asarray(d["h"])
    n = h.shape[0]
    return d, h.reshape(n, m, s, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--f", type=int, default=64)
    ap.add_argument("--n-symbols", type=int, default=14)
    ap.add_argument("--speeds", type=int, nargs="+", default=[0, 3, 6, 10],
                    help="speed 0's cube supplies the space-freq slice; every speed "
                         "(0 included) gets a time-freq slice")
    ap.add_argument("--antenna", type=int, default=0, help="fixed TX antenna index for the time-freq slice")
    ap.add_argument("--symbol", type=int, default=0, help="fixed OFDM symbol index for the space-freq slice")
    args = ap.parse_args()
    m, f, s = args.m, args.f, args.n_symbols

    if 0 in args.speeds:
        d0, cube0 = _load("train", m, s, f, 0)
        d0t, cube0t = _load("test", m, s, f, 0)
        for tag, d, cube in (("train", d0, cube0), ("test", d0t, cube0t)):
            h_sf = cube[:, :, args.symbol, :].reshape(cube.shape[0], m * f)   # (N, M, F) -> (N, M*F)
            out = f"{OUT_DIR}/{tag}_M{m}_F{f}_subregion100_ulatf.npz"
            np.savez(out, h=h_sf.astype(np.complex64), positions=d["positions"],
                     bbox=d["bbox"], m_tx=m, n_subcarriers=f, subcarrier_spacing=d["subcarrier_spacing"],
                     carrier_freq=d["carrier_freq"], n_per_position=1,
                     sampling=f"ula_tf_cube_symbol{args.symbol}_slice_of_v0")
            print(f"saved -> {out}  ({h_sf.shape[0]} x {h_sf.shape[1]})")

    for speed in args.speeds:
        dtr, cube_tr = _load("train", m, s, f, speed)
        dte, cube_te = _load("test", m, s, f, speed)
        for tag, d, cube in (("train", dtr, cube_tr), ("test", dte, cube_te)):
            h_tf = cube[:, args.antenna, :, :].reshape(cube.shape[0], s * f)   # (N, S, F) -> (N, S*F)
            out = f"{OUT_DIR}/{tag}_siso_S{s}_F{f}_v{speed}_subregion100_ulatf.npz"
            np.savez(out, h=h_tf.astype(np.complex64), positions=d["positions"],
                     velocities=d["velocities"], headings=d["headings"], speed=speed,
                     n_subcarriers=f, n_symbols=s, subcarrier_spacing=d["subcarrier_spacing"],
                     symbol_dt=d["symbol_dt"], carrier_freq=d["carrier_freq"], bbox=d["bbox"],
                     vec_order="h[t*F+f]  (OFDM-symbol-major, subcarrier-minor)",
                     sampling=f"ula_tf_cube_antenna{args.antenna}_slice")
            print(f"saved -> {out}  ({h_tf.shape[0]} x {h_tf.shape[1]})")


if __name__ == "__main__":
    main()
