

Here is the revised `guidelines.md` adapted for Jupyter Notebook-based experiments. The project structure and execution instructions have been updated so the CLI generates an `.ipynb` file for the main simulation pipeline, complete with inline visualizations.

### File: `guidelines.md`

```markdown
# Guidelines: TDD/FDD Massive MIMO-OFDM CSI Aging & Mobility Analysis

## 1. Project Overview & Objective
The goal of this project is to simulate and analyze **CSI Aging** (time-delay-induced mismatch) and **Frequency Mismatch** (FDD UL/DL frequency divergence) in high-mobility Massive MIMO-OFDM environments using the **NVIDIA Sionna** ray-tracing library.

The simulation will quantify performance degradation—specifically Normalized Mean Squared Error (NMSE) and Achievable Sum-Rate Loss—as a function of UE mobility velocity ($v$) and feedback delay ($\Delta t$). The main experiments and visualizations will be conducted interactively within a Jupyter Notebook.

---

## 2. Technical Stack & Environment Requirements
- **Python**: 3.10+
- **Primary Framework**: `sionna` (RT & PHY modules, v0.15+ or latest version compatible with TensorFlow)
- **Backend**: `tensorflow` (2.13+) with GPU support enabled
- **Scientific & Interactive Computing**: `numpy`, `scipy`, `matplotlib`, `jupyter`, `notebook`

Context
- https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html
- /home/ejseo/SNIC2026/Sionna_tutorial
- /home/ejseo/SNIC2026/example.py


---

## 3. Mathematical & System Model Specifications

### 3.1 System Parameters
- **Base Station (BS)**:
  - Antennas: $8 \times 8 = 64$ Uniform Planar Array (UPA)
  - Element spacing: $0.5\lambda$
  - Radiation pattern: `"tr38901"` or 3GPP-compliant dipole
- **User Equipment (UE)**:
  - Antennas: Single-antenna ($1 \times 1$) or $2 \times 2$ UPA
  - Velocity vector: $\mathbf{v} = [v_x, v_y, v_z]$ m/s (configurable speed up to 120 km/h)
- **OFDM Specifications**:
  - Carrier frequency ($f_c$): $3.5\text{ GHz}$ (TDD); $f_{\text{UL}} = 1.9\text{ GHz}$, $f_{\text{DL}} = 2.1\text{ GHz}$ (FDD)
  - Subcarrier Spacing (SCS): $30\text{ kHz}$
  - FFT size ($N_{\text{fft}}$): $1024$
  - Active subcarriers: $600$

### 3.2 Mobility & CSI Aging Formulation
For each ray path $i$ with initial complex gain $a_i(0)$, delay $\tau_i$, and Doppler frequency $f_{d,i} = \frac{\mathbf{v} \cdot \mathbf{\hat{k}}_i}{\lambda}$:

1. **Path Coefficient at $t = \Delta t$**:
   $$a_i(\Delta t) = a_i(0) \cdot e^{j 2\pi f_{d,i} \Delta t}$$

2. **Channel Impulse Response (CIR) to Frequency Response**:
   Convert multipath coefficients $\{a_i(t), \tau_i\}$ to subcarrier channel matrix $\mathbf{H}(t) \in \mathbb{C}^{N_{\text{sub}} \times N_{\text{rx}} \times N_{\text{tx}}}$ using Sionna's `cir_to_ofdm_channel`.

3. **CSI Mismatch Metric**:
   $$\text{NMSE}(\Delta t) = \frac{\mathbb{E} \left[ \Vert{}\mathbf{H}(t + \Delta t) - \mathbf{H}(t)\Vert{}_F^2 \right]}{\mathbb{E} \left[ \Vert{}\mathbf{H}(t)\Vert{}_F^2 \right]}$$

---

## 4. Required Project Structure

Implement the project using a modular Python package layout, utilizing a Jupyter Notebook for the top-level execution:

```text
csi_mobility_analysis/
├── config.py                  # Global parameters, frequencies, array configs
├── scene_builder.py           # Sionna RT scene initialization, BS/UE positioning
├── channel_engine.py          # Ray tracing, Doppler phase rotation, OFDM conversion
├── precoder.py                # MRT, Zero-Forcing (ZF), and MMSE precoding implementations
├── evaluator.py               # NMSE, SINR, and Achievable Rate calculation logic
├── experiments/
│   └── mobility_analysis.ipynb # Main Jupyter notebook for parameter sweeps and inline plotting
└── tests/
    └── test_channel.py        # Unit tests for Doppler rotation & channel shapes

```

---

## 5. Core Implementation Details

### `config.py`

Define a structured dataclass `SimulationConfig`:

* `fc`: Carrier frequency (Hz)
* `scs`: Subcarrier spacing
* `fft_size`: FFT size
* `ue_velocity`: 3D tuple `(v_x, v_y, v_z)` in m/s
* `delta_t_list`: List of feedback delay intervals (e.g., `[0, 1e-3, 2e-3, 5e-3, 10e-3]`)

### `channel_engine.py`

* Compute ray paths using `scene.compute_paths(max_depth=3)`.
* Extract `paths.a`, `paths.tau`, and `paths.doppler`.
* Implement a method `get_channel_at_time(delta_t)` that applies the exponential phase shift factor $\exp(j 2\pi \cdot f_d \cdot \Delta t)$ to `paths.a` before converting to frequency domain.

### `precoder.py`

* Implement Zero-Forcing (ZF) precoder based on estimated CSI $\mathbf{\hat{H}} = \mathbf{H}(t)$:

$$\mathbf{W}_{\text{ZF}} = \mathbf{\hat{H}}^H \left( \mathbf{\hat{H}} \mathbf{\hat{H}}^H \right)^{-1}$$


* Normalize power across transmitting antennas.

### `experiments/mobility_analysis.ipynb`

* **Cell 1**: Imports, system path configuration, and TensorFlow GPU setup.
* **Cell 2**: Markdown explaining the simulation scenario.
* **Cell 3**: Scene initialization and RT computation (using `scene_builder.py` and `channel_engine.py`).
* **Cell 4**: Execution of the parameter sweep (iterating over velocities and $\Delta t$).
* **Cell 5**: Matplotlib inline plotting for NMSE and Achievable Rate results.

---

## 6. Expected Output & Artifacts

Running all cells in `experiments/mobility_analysis.ipynb` must output inline visual graphs:

1. **NMSE vs. Delay**: NMSE (dB) plotted against Feedback Delay $\Delta t$ for varying UE velocities ($v = 15, 30, 60 \text{ km/h}$).
2. **Sum-Rate Degradation**: Achievable Rate (bps/Hz) plotted against Feedback Delay $\Delta t$, demonstrating capacity decay due to CSI Aging.

---

## 7. Development Rules for Antigravity

1. **No Placeholder Code**: Write complete, executable Python code and fully formed `.ipynb` JSON structures. Do not use `pass` or `# TODO`.
2. **Notebook Format Validation**: When generating the `.ipynb` file, ensure it follows valid Jupyter Notebook JSON schema (v4).
3. **GPU Memory Growth**: Include code to explicitly enable TensorFlow GPU memory growth in the first cell of the notebook.
4. **Shape Integrity Check**: Explicitly log and verify tensor shapes when bridging ray-tracing path outputs to OFDM channel matrices within the notebook execution.

---

