import json

notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_code_cell(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    })

def add_md_cell(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    })

# Cell 1
add_code_cell("""import sys
import os
sys.path.append(os.path.abspath('..'))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# GPU Memory Growth Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)
        
from config import Config
from scene_engine import setup_scene
from channel_analyzer import compute_batch_paths, get_ofdm_channel, compute_spatial_covariance
from dataset_exporter import save_dataset
""")

# Cell 2
add_code_cell("""config = Config()
scene, bs, ue, pos_tensor = setup_scene(config)
print("Trajectory shape:", pos_tensor.shape)
""")

# Cell 3
add_code_cell("""paths = compute_batch_paths(scene, max_depth=3)
h_tensor = get_ofdm_channel(paths, config)
print("H tensor shape (N_pos, N_sub, N_tx):", h_tensor.shape)
""")

# Cell 4
add_code_cell("""R_matrix = compute_spatial_covariance(h_tensor)
print("Spatial Covariance Matrix shape (N_sub, N_tx, N_tx):", R_matrix.shape)
# Average across subcarriers for visualization
R_avg = tf.reduce_mean(R_matrix, axis=0)
print("Averaged R shape:", R_avg.shape)
""")

# Cell 5
add_code_cell("""plt.figure(figsize=(8, 6))
plt.imshow(np.abs(R_avg.numpy()), cmap='viridis')
plt.colorbar(label='Magnitude')
plt.title('Spatial Covariance Matrix |R|')
plt.xlabel('Tx Antenna Index')
plt.ylabel('Tx Antenna Index')
plt.show()

# Export data
save_dataset(h_tensor, R_matrix)
""")

with open('/home/ejseo/SNIC2026/experiments/covariance_visualization.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated.")
