# Sionna-based CSI Aging and Mobility Simulation

## 1. Project Context Brief
This project implements a comprehensive simulation environment to analyze **CSI (Channel State Information) Aging** and **Frequency Mismatch** in high-mobility, Massive MIMO-OFDM communication systems. Built on the **NVIDIA Sionna** ray-tracing and PHY layer library, the simulation accurately models Doppler-induced phase shifts over arbitrary feedback delays ($\Delta t$) and mobile velocities ($v$) in complex scattering environments (e.g., 3D street canyons).

## 2. Current Focus & Constraints
**Focus**: Quantify performance degradation metrics—specifically Normalized Mean Squared Error (NMSE) and Achievable Sum-Rate Loss—as a function of User Equipment (UE) mobility velocity and channel feedback delay in TDD scenarios. 

**Constraints**:
- **Framework**: NVIDIA Sionna v0.15+ / TensorFlow 2.13+ with GPU acceleration.
- **Topology**: Base Station (64-antenna UPA), Single-antenna UE.
- **Spectrum**: 3.5 GHz (TDD) carrier frequency, 30 kHz subcarrier spacing, 1024-FFT, 600 active subcarriers.
- **Execution**: Modular Python architecture governed by a central Jupyter Notebook (`experiments/mobility_analysis.ipynb`) handling GPU memory growth and inline Matplotlib visualization.

## 3. Core Architecture
The repository utilizes a modular structure for isolating simulation components:
- `config.py`: Global simulation parameters (frequencies, array dimensions, velocities, and delay sweeps).
- `scene_builder.py`: Initializes the Sionna RT scene (`etoile` environment) and instantiates Transmitter/Receiver planar arrays.
- `channel_engine.py`: Handles high-fidelity RT path computation via `sionna.rt.PathSolver` and extracts time-evolved OFDM channel matrices using Doppler shift phase evolution via `.cfr()`.
- `precoder.py`: Computes Zero-Forcing (ZF) precoding matrices based on estimated CSI.
- `evaluator.py`: Calculates NMSE and Achievable Sum-Rate given the true and estimated channels.

## 4. Data Pipeline & Optimization
- **TensorFlow Execution**: All channel matrices, precoders, and metric evaluations are performed using deeply nested complex-valued TensorFlow tensors for massive parallelism.
- **GPU Memory Growth**: Ensures graceful allocation on arbitrary NVIDIA GPUs, configured during Jupyter Notebook initialization to prevent Out-Of-Memory (OOM) failures.
- **Time Evolution (`cfr`)**: Instead of manually reconstructing frequency-domain responses from DrJit properties, the pipeline optimally leverages Sionna's native Channel Frequency Response (`cfr`) generation with variable sampling rates to perfectly emulate arbitrary discrete delays ($\Delta t$).

## 5. Instructions
To execute the simulation:
1. Ensure the Python environment has `tensorflow`, `sionna`, `jupyter`, and `matplotlib` installed.
2. Launch Jupyter Notebook or Jupyter Lab from the project root.
3. Open `experiments/mobility_analysis.ipynb`.
4. Select "Run All Cells" to execute the simulation, compute sweeps, and render visualizations.

## 6. Outline
```text
SNIC2026/
├── config.py
├── scene_builder.py
├── channel_engine.py
├── precoder.py
├── evaluator.py
├── make_notebook.py
├── experiment_log.md
├── experiments/
│   └── mobility_analysis.ipynb
└── tests/
    └── test_channel.py
```

## 7. Code Execution
The full pipeline can be statically generated and executed via CLI:
```bash
python make_notebook.py
jupyter nbconvert --execute --to notebook --inplace experiments/mobility_analysis.ipynb
```
*Note: This generates the Notebook JSON and natively computes all TensorFlow paths, embedding resulting plots back into the `.ipynb`.*

## 8. Descriptions
- **NMSE vs Feedback Delay**: Measures the divergence of the channel between estimation ($t=0$) and data transmission ($t=\Delta t$).
- **Sum-Rate Degradation**: Translates the imperfect Zero-Forcing cancellation from outdated CSI into a physical capacity penalty (bps/Hz).
- **Scene and Path Renders**: Interactive and static plots leveraging Sionna's internal rendering engines to validate line-of-sight and non-line-of-sight propagation geometries.

## 9. LLM Agent Contributions
- **Component Design**: Developed the modular object-oriented simulation layers (`config`, `scene_builder`, `channel_engine`, `precoder`, `evaluator`).
- **Jupyter Notebook Serialization**: Wrote an automated Python-to-JSON serialization script (`make_notebook.py`) to generate a perfectly valid Jupyter Notebook (schema v4).
- **Algorithm Migration**: Re-architected CSI aging logic to comply with modern Sionna (v0.15+) standards, safely translating DrJit tuples to TF tensors via `PathSolver` and the `cfr` function.
- **Interactive Visuals**: Retrofitted the simulation with interactive path visualization and static 2D renders directly embedded within the final executed Jupyter output.