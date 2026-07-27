
# Guidelines: Trajectory-Based Spatial Covariance Analysis in Massive MIMO

## 1. Project Overview & Objective
The objective is to simulate and analyze the Spatial Covariance Matrix ($\mathbf{R}$) of a Massive MIMO channel under user mobility using the **NVIDIA Sionna** Ray Tracing (RT) module. 

Instead of re-computing ray tracing iteratively, the simulation must leverage TensorFlow's batching capabilities to compute paths across a dense physical trajectory simultaneously. The output must quantify spatial correlation and format the resulting channel tensors for downstream deep learning tasks (e.g., CSI feedback and compression).

---

## 2. Technical Stack & Environment Requirements
- **Python**: 3.10+
- **Primary Framework**: `sionna` (RT & PHY modules)
- **Backend**: `tensorflow` (2.13+) with GPU memory growth enabled.
- **Scientific Computing**: `numpy`, `matplotlib`, `jupyter`

---

## 3. Mathematical & System Model Specifications

### 3.1 System Parameters
- **Base Station (BS)**:
  - Antennas: $8 \times 8 = 64$ Uniform Planar Array (UPA)
  - Element spacing: $0.5\lambda$
  - Pattern/Polarization: `"tr38901"`, Vertical (`"V"`)
- **User Equipment (UE)**:
  - Antennas: Single-antenna ($1 \times 1$) dipole.
  - Trajectory: A batch tensor of 3D coordinates representing a linear path (e.g., $N_{\text{pos}} = 50$ points spaced by 1 meter).
  - Velocity vector: $\mathbf{v} = [0, 15, 0]$ m/s (used for small-scale Doppler calculations).
- **OFDM Specifications**:
  - Carrier frequency ($f_c$): $3.5\text{ GHz}$
  - Subcarrier Spacing (SCS): $30\text{ kHz}$
  - Active subcarriers: $600$

### 3.2 Channel and Covariance Formulation
1. **Parallel Ray Tracing**:
   Extract $\mathbf{a}$ (path gains) and $\mathbf{\tau}$ (delays) for the entire batch $N_{\text{pos}}$.
2. **Frequency Domain Conversion**:
   Convert the multipath CIR to subcarrier channel matrices $\mathbf{H} \in \mathbb{C}^{N_{\text{pos}} \times N_{\text{sub}} \times N_{\text{tx}}}$.
3. **Spatial Covariance Matrix ($\mathbf{R}$)**:
   For a specific subcarrier (or averaged across subcarriers), calculate the covariance matrix across the trajectory:
   $$\mathbf{R} = \frac{1}{N_{\text{pos}}} \sum_{i=1}^{N_{\text{pos}}} \mathbf{h}_i \mathbf{h}_i^H$$
   where $\mathbf{h}_i \in \mathbb{C}^{N_{\text{tx}} \times 1}$ is the channel vector at trajectory point $i$.

---

## 4. Required Project Structure

```text
spatial_covariance_project/
├── config.py                  # Global parameters and array configurations
├── scene_engine.py            # Sionna RT scene setup, trajectory tensor generation
├── channel_analyzer.py        # Batch path computation, OFDM conversion, covariance math
├── dataset_exporter.py        # Utility to save H tensors and R matrices as .npy files
└── experiments/
    └── covariance_visualization.ipynb # Main Jupyter notebook for execution and plotting

```

---

## 5. Core Implementation Details

### `scene_engine.py`

* Load `sionna.rt.scene.etoile`.
* Generate a trajectory batch tensor: `tf.stack([x_pos, y_pos, z_pos], axis=1)` with shape `[N_pos, 3]`.
* Assign the trajectory directly to the `position` argument of the `Receiver` object.

### `channel_analyzer.py`

* Compute paths using `scene.compute_paths(max_depth=3)`.
* Use `cir_to_ofdm_channel` to get the frequency response. Squeeze the dimension corresponding to the single RX antenna to yield shape `[N_pos, N_sub, N_tx]`.
* Implement `compute_spatial_covariance(h_tensor)` using `tf.matmul` with `adjoint_b=True` to compute the outer product $\mathbf{h}\mathbf{h}^H$ for each position, then use `tf.reduce_mean(..., axis=0)` to yield the final $64 \times 64$ matrix.

### `dataset_exporter.py`

* Provide functions to save the generated OFDM channel tensor and the Covariance matrix to disk using `np.save()`, ensuring they are ready for neural network ingestion.

### `experiments/covariance_visualization.ipynb`

* **Cell 1**: Standard imports and TensorFlow GPU memory growth setup.
* **Cell 2**: Initialize the scene and trajectory.
* **Cell 3**: Compute the paths and OFDM channel.
* **Cell 4**: Calculate the spatial covariance matrix $\mathbf{R}$.
* **Cell 5**: Plot the absolute values of the covariance matrix using `plt.imshow(np.abs(R_mean))`. Add a colorbar and appropriate labels (Tx Antenna Index).

---

## 6. Development Rules for Antigravity

1. **Strictly Executable**: Generate complete Python modules and a valid Jupyter Notebook (JSON schema v4) with no placeholder code (`pass` or `# TODO`).
2. **Tensor Shapes**: Maintain strict discipline over tensor dimensions. Log shapes explicitly in the notebook cells before performing `tf.matmul` operations.
3. **GPU Check**: Include code to verify and enable GPU memory growth to prevent Out-Of-Memory (OOM) errors during the batch trajectory computation.


---
