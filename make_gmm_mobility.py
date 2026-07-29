import json

cells = []

def add_code_cell(source_code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_code.split('\n')]
    })

# Setup Cell
add_code_cell("""
import os, sys
import tensorflow as tf
import numpy as np
import torch
import matplotlib.pyplot as plt

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import sys, os
sys.path.append(os.path.abspath('..'))

from config import Config
from scene_engine import setup_scene, setup_scene_with_positions
from channel_analyzer import compute_batch_paths, get_ofdm_channel
from GMM.gmtc_em import gmm_em_fit_fullcov, predict_modes
config = Config()
""")

# Cell A1: Grid generation
add_code_cell("""
# Cell A1: Dense spatial grid generation
x_range = np.linspace(20, 80, 15)
y_range = np.linspace(-10, 70, 20)
grid_x, grid_y = np.meshgrid(x_range, y_range)
z_fixed = 1.5

grid_positions = np.stack(
    [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z_fixed)], axis=-1
)
print("Grid map positions:", grid_positions.shape)
""")

# Cell A2: Grid channel extraction
add_code_cell("""
# Cell A2: Channel extraction for grid
scene_map, bs_map, _, _ = setup_scene_with_positions(config, grid_positions)
paths_map = compute_batch_paths(scene_map, max_depth=2)
h_grid = get_ofdm_channel(paths_map, config)
print("Grid channel shape:", h_grid.shape)
""")

# Cell A3: GMM Training
add_code_cell("""
# Cell A3: Train GMM on grid data
N_grid, N_sub, N_tx = h_grid.shape
sub_stride = 10
sub_idx = tf.range(0, N_sub, sub_stride)
h_map_train = tf.reshape(tf.gather(h_grid, sub_idx, axis=1), [-1, N_tx])

h_map_real = tf.concat([tf.math.real(h_map_train), tf.math.imag(h_map_train)], axis=-1)
Xt_map_raw = torch.from_numpy(h_map_real.numpy()).to(torch.float32)
Xt_map = torch.nan_to_num(Xt_map_raw, nan=0.0)

print(f"Training GMM with {Xt_map.shape[0]} samples...")
print(f"Contains NaN: {torch.isnan(Xt_map).any().item()}")
print(f"Max val: {Xt_map.max().item()}, Min val: {Xt_map.min().item()}")
K = 4
params_map = gmm_em_fit_fullcov(Xt_map, K=K, num_iter=100, eps=1e-5, seed=42, verbose=True)

import os
os.makedirs("output", exist_ok=True)
torch.save(params_map, f"output/gmm_spatial_dictionary_K{K}.pt")
""")

# Cell A4: EM Map Visualization
add_code_cell("""
# Cell A4: Spatial EM Map visualization
sub_stride = 10
sub_idx = tf.range(0, N_sub, sub_stride)

h_grid_sub = tf.gather(h_grid, sub_idx, axis=1) # [N_grid, N_sub_sub, N_tx]
N_sub_sub = h_grid_sub.shape[1]

h_grid_sub_real = tf.concat([tf.math.real(h_grid_sub), tf.math.imag(h_grid_sub)], axis=-1)
h_grid_sub_torch = torch.nan_to_num(torch.from_numpy(h_grid_sub_real.numpy()).to(torch.float32), nan=0.0)

c_hat_grid_all = np.zeros((N_grid, N_sub_sub), dtype=int)
for i in range(N_sub_sub):
    c, _ = predict_modes(h_grid_sub_torch[:, i, :], params_map)
    c_hat_grid_all[:, i] = c.numpy()

from scipy.stats import mode
c_hat_grid, _ = mode(c_hat_grid_all, axis=1, keepdims=False)

cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(7, 7))
for k in range(K):
    m = (c_hat_grid == k)
    ax.scatter(grid_positions[m, 0], grid_positions[m, 1], s=30, color=cmap(k), label=f"State {k}")
ax.scatter(config.pos_start[0], config.pos_start[1], marker="*", s=300, c="red", edgecolor="k", label="Start Pos")
ax.set_title(f"Spatial GMM Cluster Map (K={K})")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_aspect("equal")
plt.savefig(f"output/em_map_spatial_K{K}.png", dpi=120, bbox_inches='tight')
plt.show()
""")

# Cell B1: Trajectory channel
add_code_cell("""
# Cell B1: Load dictionary and query for trajectory
scene_traj, bs_traj, _, pos_tensor = setup_scene(config)
paths_traj = compute_batch_paths(scene_traj, max_depth=3)
h_traj = get_ofdm_channel(paths_traj, config)
print("Trajectory channel shape:", h_traj.shape)

h_traj_sub = tf.gather(h_traj, sub_idx, axis=1)
h_traj_sub_real = tf.concat([tf.math.real(h_traj_sub), tf.math.imag(h_traj_sub)], axis=-1)
h_traj_sub_torch = torch.nan_to_num(torch.from_numpy(h_traj_sub_real.numpy()).to(torch.float32), nan=0.0)

params_map = torch.load(f"output/gmm_spatial_dictionary_K{K}.pt", weights_only=False)

N_traj = h_traj.shape[0]
c_hat_traj_all = np.zeros((N_traj, N_sub_sub), dtype=int)
for i in range(N_sub_sub):
    c, _ = predict_modes(h_traj_sub_torch[:, i, :], params_map)
    c_hat_traj_all[:, i] = c.numpy()

c_hat_traj, _ = mode(c_hat_traj_all, axis=1, keepdims=False)

from collections import Counter
counts = Counter(c_hat_traj)
print("Trajectory State Counts:", counts)
""")

# Cell B2: Trajectory overlaid map
add_code_cell("""
# Cell B2: Trajectory-overlaid EM map
fig, ax = plt.subplots(figsize=(7, 7))
for k in range(K):
    m = (c_hat_grid == k)
    ax.scatter(grid_positions[m, 0], grid_positions[m, 1], s=30, alpha=0.3, color=cmap(k))

traj_np = pos_tensor.numpy()
ax.plot(traj_np[:, 0], traj_np[:, 1], 'k-', label="Trajectory")
for k in range(K):
    m = (c_hat_traj == k)
    if np.any(m):
        ax.scatter(traj_np[m, 0], traj_np[m, 1], s=50, color=cmap(k), edgecolor='k', label=f"Traj State {k}")

ax.set_title(f"Trajectory Overlaid on EM Map (K={K})")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_aspect("equal")
plt.savefig(f"output/em_map_with_trajectory_K{K}.png", dpi=120, bbox_inches='tight')
plt.show()
""")

# Cell B3: GMTC Compression & NMSE
add_code_cell("""
# Cell B3: Re-validate GMTC Compression Pipeline
def reconstruct_complex_cov(R_real):
    N_t = R_real.shape[0] // 2
    R_xx = R_real[:N_t, :N_t]
    R_xy = R_real[:N_t, N_t:]
    return 2 * (R_xx - 1j * R_xy)

U_c_complex_list = []
lam_c_complex_list = []

for c in range(K):
    U_c_real = params_map.U[c].numpy()
    lam_c_real = params_map.lam[c].numpy()
    R_c_real = U_c_real @ np.diag(lam_c_real) @ U_c_real.T
    R_c_comp = reconstruct_complex_cov(R_c_real)
    
    evals, evecs = np.linalg.eigh(R_c_comp)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    lam_c_complex_list.append(evals)
    U_c_complex_list.append(evecs)

h_spatial_np = np.nan_to_num(h_traj[:, 300, :].numpy(), nan=0.0)
N_pos = h_spatial_np.shape[0]
h_hat = np.zeros_like(h_spatial_np)

mu = 0.05

for t in range(N_pos):
    c_hat_t = c_hat_traj[t].item()
    h_t = h_spatial_np[t]
    
    U_c = U_c_complex_list[c_hat_t]
    lam_c = lam_c_complex_list[c_hat_t]
    
    h_tilde = U_c.conj().T @ h_t
    mask = lam_c > mu
    h_tilde_q = h_tilde * mask
    
    h_hat[t] = U_c @ h_tilde_q

nmse_linear = np.sum(np.abs(h_spatial_np - h_hat)**2, axis=1) / np.sum(np.abs(h_spatial_np)**2, axis=1)
nmse_db = 10 * np.log10(nmse_linear + 1e-12)

plt.figure(figsize=(10, 5))
plt.plot(range(N_pos), nmse_db, marker="o", linestyle="-")
plt.title(f"NMSE of GMTC Reconstruction across Trajectory (mu={mu})")
plt.xlabel("Trajectory Step (pos)")
plt.ylabel("NMSE (dB)")
plt.grid(True)
plt.show()

print(f"Average NMSE (dB): {np.mean(nmse_db):.2f}")
""")

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("experiments/GMM_mobility.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
