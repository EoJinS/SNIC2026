# Sionna-based CSI Aging and Spatial Covariance Analysis

## 1. Project Context Brief
This project implements a comprehensive simulation environment using the **NVIDIA Sionna** ray-tracing and PHY layer library to analyze:
1. **CSI (Channel State Information) Aging and Frequency Mismatch** in high-mobility, Massive MIMO-OFDM communication systems.
2. **Spatial Covariance Analysis** under user mobility. By leveraging trajectory batching, the simulation accurately extracts multipath propagation characteristics and computes spatial correlation matrices ($\mathbf{R}$) optimized for downstream deep learning workflows (e.g., CSI feedback compression autoencoders).

## 2. Current Focus & Constraints
**Focus**: 
- Quantify performance degradation metrics (NMSE and Achievable Sum-Rate Loss) over feedback delays and velocity.
- Extract high-fidelity spatial covariance matrices across a continuous user trajectory for ML dataset generation.

**Constraints**:
- **Framework**: NVIDIA Sionna v0.15+ / TensorFlow 2.13+ with GPU acceleration.
- **Topology**: Base Station (8x8 UPA, 64-antenna, $0.5\lambda$ spacing, Vertical polarization), Single-antenna UE (1x1 dipole).
- **Trajectory**: Batch tensor of 3D coordinates representing a linear path ($N_{\text{pos}} = 50$ points spaced by 1 meter) with a velocity vector $\mathbf{v} = [0, 15, 0]$ m/s.
- **Spectrum**: 3.5 GHz (TDD) carrier frequency, 30 kHz subcarrier spacing, 1024-FFT, 600 active subcarriers.
- **Execution**: Modular Python architecture governed by central Jupyter Notebooks that handle GPU memory growth, inline Matplotlib visualization, and tensor exports.

## 3. Core Architecture
The repository utilizes a modular structure for isolating simulation components:
- `config.py`: Global simulation parameters (frequencies, array dimensions, velocities, and delay sweeps).
- `scene_builder.py` & `scene_engine.py`: Initializes the Sionna RT scene (`etoile` environment), instantiates arrays, and generates trajectory batch receivers.
- `channel_engine.py` & `channel_analyzer.py`: Handles RT path computation (max_depth=3) via `sionna.rt.PathSolver`, extracts time-evolved OFDM channel matrices ($\mathbf{H} \in \mathbb{C}^{N_{\text{pos}} \times N_{\text{sub}} \times N_{\text{tx}}}$), and calculates spatial covariance matrices ($\mathbf{R} = \frac{1}{N_{\text{pos}}} \sum_{i=1}^{N_{\text{pos}}} \mathbf{h}_i \mathbf{h}_i^H$) using massive outer-product tensor operations.
- `precoder.py` & `evaluator.py`: Computes Zero-Forcing (ZF) precoding matrices and evaluates NMSE and Achievable Sum-Rate.
- `dataset_exporter.py`: Utility to export generated OFDM tensors ($\mathbf{H}$) and covariance matrices ($\mathbf{R}$) as `.npy` files for ML pipelines.

## 4. Data Pipeline & Optimization
- **TensorFlow Execution**: All channel matrices, precoders, and metric evaluations are performed using deeply nested complex-valued TensorFlow tensors for massive parallelism.
- **Batch Trajectory Ray Tracing**: Exploits parallel receiver simulations in Sionna's PathSolver to evaluate continuous user trajectories instantaneously. Tensor dimensions are strictly maintained and logged.
- **Spatial Covariance Math**: Computes the outer product $\mathbf{h}\mathbf{h}^H$ using `tf.matmul(..., adjoint_b=True)` and averages over the trajectory using `tf.reduce_mean(..., axis=0)`.
- **GPU Memory Growth**: Ensures graceful allocation on arbitrary NVIDIA GPUs, configured during Jupyter Notebook initialization to prevent Out-Of-Memory (OOM) failures.

## 5. Instructions
To execute the simulation:
1. Ensure the Python environment has `tensorflow`, `sionna`, `jupyter`, and `matplotlib` installed.
2. Launch Jupyter Notebook or Jupyter Lab from the project root.
3. Open `experiments/covariance_visualization.ipynb`.
4. Select "Run All Cells" to execute the simulation, compute sweeps, and render visualizations.

## 6. Outline
```text
SNIC2026/
├── config.py
├── scene_builder.py
├── scene_engine.py
├── channel_engine.py
├── channel_analyzer.py
├── precoder.py
├── evaluator.py
├── dataset_exporter.py
├── data/
│   ├── h_tensor.npy
│   └── R_matrix.npy
├── experiments/
│   ├── mobility_example.ipynb
│   └── covariance_visualization.ipynb
└── tests/
    └── test_channel.py
```

## 7. Code Execution
The full pipelines can be statically generated and executed via CLI:
```bash
# CSI Aging Pipeline
python make_notebook.py
jupyter nbconvert --execute --to notebook --inplace experiments/mobility_example.ipynb

# Spatial Covariance Pipeline
python make_notebook2.py
jupyter nbconvert --execute --to notebook --inplace experiments/covariance_visualization.ipynb
```

## 8. Descriptions
- **NMSE vs Feedback Delay**: Measures the divergence of the channel between estimation ($t=0$) and data transmission ($t=\Delta t$).
- **Sum-Rate Degradation**: Translates the imperfect Zero-Forcing cancellation from outdated CSI into a physical capacity penalty.
- **Spatial Covariance Matrix**: An aggregated visualization (e.g., using `plt.imshow` with the `viridis` colormap on absolute values) of the $64 \times 64$ cross-correlation across transmitting antenna elements for a mobile user trajectory, averaged across subcarriers.
- **Scene and Path Renders**: Interactive and static plots leveraging Sionna's internal rendering engines to validate line-of-sight and non-line-of-sight propagation geometries.

## 9. LLM Agent Contributions
- **Component Design**: Developed the modular object-oriented simulation layers (`config`, `scene_engine`, `channel_analyzer`, `dataset_exporter`).
- **Jupyter Notebook Serialization**: Wrote automated Python-to-JSON serialization scripts (`make_notebook.py`, `make_notebook2.py`) to generate perfectly valid Jupyter Notebooks (schema v4).
- **Parallel Trajectory Modeling**: Overcame Sionna versioning constraints by strategically instantiating parallel `Receiver` agents along a trajectory to yield synchronized CFR tensors without OOM errors.
- **Machine Learning Integration**: Exported generated datasets directly into `.npy` formats, formatting dimensions exactly to specification (`[N_sub, N_tx, N_tx]`) for direct integration into neural network training loops.

## 10. Modifications to Guideline
During the integration of the GMM module in `covariance_visualization.ipynb`, the following modifications were made to `guideline.md` to ensure correct functionality:
- **GMM Dictionary Training**: Since a pre-trained dictionary was not provided, an EM fitting step (`gmm_em_fit_fullcov`) was added to train the GMM dictionary on the generated CSI dataset dynamically.
- **Complex-to-Real Mapping**: The provided GMM module processes real-valued data, while the CSI vector is complex. The mathematical formulation for the log-likelihood MAP inference was updated to convert the complex CSI vector $h \in \mathbb{C}^{N_t}$ to its real isomorphic counterpart $h_{real} \in \mathbb{R}^{2N_t}$, concatenating its real and imaginary parts.
- **Spatial Vectorization**: Vectorizing the $N_t \times N_c$ matrix into a $38400 \times 1$ vector was aborted as it would result in Out-Of-Memory (OOM) errors during full-covariance EM fitting. The implementation was adjusted to extract the spatial CSI vector $h \in \mathbb{C}^{N_t}$ (by selecting the center subcarrier) to maintain a manageable dimension for the $64 \times 64$ spatial covariance matrices.
- **Two-Stage Architecture Refactoring**: Refactored `GMM_mobility.ipynb` to completely decouple EM dictionary map building (Stage A) from query-only trajectory handling (Stage B). This involved generalizing the Sionna scene builder, implementing multi-subcarrier aggregation for dense grid map training (18,000 samples for K=4), enforcing "majority vote" classification per physical coordinate across OFDM spectra, and plotting comprehensive spatial EM cluster maps overlaid with mobile user trajectories. The resulting well-estimated pre-trained dictionary lowered the average NMSE of the GMTC compression pipeline across the mobile trajectory from -14.32 dB to a highly optimal -31.22 dB.
- **Covariance Reconstruction Scaling**: A mathematical bug was identified where rebuilding the complex covariance matrix from the learned real components ($R_{xx}$, $R_{xy}$) resulted in half-scaled magnitudes $\frac{1}{2}|R_c|$. The matrix reconstruction function in the notebook was updated to accurately scale and conjugate the blocks via `2 * (R_xx - 1j * R_xy)`, aligning perfectly with the theoretical expectation for proper circularly symmetric complex Gaussian covariances.
- **GMTC Compression and NMSE Evaluation**: Added the final stages of the GMTC pipeline into the notebook. For each trajectory point, the spatial CSI vector is transformed using the component-matched Karhunen-Loève Transform (KLT) based on the complex eigenvectors of the active GMM MAP state. Truncation is simulated via a reverse-waterfilling tunable threshold ($\mu = 0.05$), and the channel is reconstructed (Inverse KLT) to compute and plot the Normalized Mean Squared Error (NMSE) across the user's continuous trajectory.