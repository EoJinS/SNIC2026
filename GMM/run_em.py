#!/usr/bin/env python
"""
run_em.py — ASU 채널(32개마다 뽑은 32 서브캐리어)로 GMTC EM 클러스터링 → 맵
===========================================================================
파이프라인:
  H(T,32,32) → real∥imag concat (T,2048) → N=256 청크(유저당 8개)
  → GMTC full-cov GMM EM → 청크 라벨 → 유저 라벨 = 8청크 다수결 → 맵 PNG

실행:
  python run_em.py                      # 기본: data/asu_fd_32x32_stride32.npy, K=4,8,16
  python run_em.py --Ks 8 16 32         # K 지정
  CUDA_VISIBLE_DEVICES=0 python run_em.py   # GPU 사용(없으면 자동 CPU)

의존: torch, numpy, matplotlib  (같은 폴더의 gmtc_em.py 사용, 외부 경로 불필요)
"""
import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from gmtc_em import gmm_em_fit_fullcov, predict_modes   # 같은 폴더의 자립 모듈

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "data/asu_fd_32x32_stride32.npy"))
    p.add_argument("--pos", default=os.path.join(HERE, "data/asu_pos.npz"))
    p.add_argument("--N", type=int, default=256, help="GMTC 청크 크기")
    p.add_argument("--Ks", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--em_iter", type=int, default=50)
    p.add_argument("--outdir", default=os.path.join(HERE, "output"))
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    # 채널 → 청크 (GMTC와 동일)
    H = np.asarray(np.load(args.data, mmap_mode="r")[:, 0]).astype(np.complex64)   # (T,32,32)
    T = H.shape[0]
    real = np.concatenate([H.real.reshape(T, -1), H.imag.reshape(T, -1)], 1)       # (T,2048)
    N = args.N
    cpc = real.shape[1] // N                                                        # 유저당 청크 수(=8)
    chunks = real[:, :cpc * N].reshape(-1, N).astype(np.float32)                     # (T*cpc, 256)
    print(f"device={dev}  channel {H.shape} -> chunks {chunks.shape} ({cpc}/user)")
    Xt = torch.from_numpy(chunks)

    pz = np.load(args.pos); pos = pz["pos"]; tx = pz["tx"]
    pos_nouser = pz["pos_nouser"] if "pos_nouser" in pz else None
    assert len(pos) == T, f"pos({len(pos)}) != users({T})"
    cmap = plt.get_cmap("tab20")
    dtag = os.path.splitext(os.path.basename(args.data))[0]

    for K in args.Ks:
        params = gmm_em_fit_fullcov(Xt, K=K, num_iter=args.em_iter, seed=0, verbose=False, device=dev)
        c_hat, _ = predict_modes(Xt.to(dev), params)
        clab = c_hat.cpu().numpy()                                   # 청크 라벨
        ulab = np.array([np.bincount(clab[i * cpc:(i + 1) * cpc], minlength=K).argmax()
                         for i in range(T)])                         # 유저 라벨(다수결)

        fig, ax = plt.subplots(figsize=(8.5, 7))
        handles = []
        if pos_nouser is not None:
            ax.scatter(pos_nouser[:, 0], pos_nouser[:, 1], s=3, c="lightgray")
            handles.append(Line2D([], [], marker="o", ls="", ms=7, color="lightgray", label="No user"))
        for k in range(K):
            m = ulab == k
            if m.sum():
                ax.scatter(pos[m, 0], pos[m, 1], s=6, color=cmap(k % 20))
                handles.append(Line2D([], [], marker="o", ls="", ms=7, color=cmap(k % 20), label=f"Label {k+1}"))
        ax.scatter(tx[0], tx[1], marker="*", s=480, c="red", edgecolor="k", zorder=5)
        handles.append(Line2D([], [], marker="*", ls="", ms=13, color="red", mec="k", label="BS"))
        ax.set(title=f"ASU GMTC EM clustering (N={N}, majority vote)  K={K}",
               xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal"); ax.grid(alpha=.2)
        ax.legend(handles=handles, fontsize=8, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), framealpha=.9)
        plt.tight_layout()
        out = os.path.join(args.outdir, f"em_map_{dtag}_K{K}.png")
        plt.savefig(out, dpi=120); plt.close()
        print(f"saved {out}")


if __name__ == "__main__":
    main()
