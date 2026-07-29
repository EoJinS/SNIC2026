
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



# Cell A1: Dense spatial grid generation
x_range = np.linspace(20, 80, 15)
y_range = np.linspace(-10, 70, 20)
grid_x, grid_y = np.meshgrid(x_range, y_range)
z_fixed = 1.5

grid_positions = np.stack(
    [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z_fixed)], axis=-1
)
print("Grid map positions:", grid_positions.shape)



# Cell A2: Channel extraction for grid
scene_map, bs_map, _, _ = setup_scene_with_positions(config, grid_positions)
paths_map = compute_batch_paths(scene_map, max_depth=2)
h_grid = get_ofdm_channel(paths_map, config)
print("Grid channel shape:", h_grid.shape)



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


