# Experiment Log

## Sionna-based CSI Aging and Mobility Simulation

**Objective**: Implement a complete simulation for CSI Aging and Frequency Mismatch in high-mobility Massive MIMO-OFDM environments using the NVIDIA Sionna library.

**Implementation Steps Completed**:
1. Developed underlying Python modules (`config.py`, `scene_builder.py`, `channel_engine.py`, `precoder.py`, `evaluator.py`, `tests/test_channel.py`) in the `/home/ejseo/SNIC2026/` directory.
2. Built a script to programmatically construct a valid Jupyter Notebook JSON document (schema v4).
3. The generated notebook (`experiments/mobility_analysis.ipynb`) effectively:
   - Configures system and handles TensorFlow GPU memory growth appropriately in its first cell.
   - Computes ray paths in a 3D street canyon (or generic) scene using `sionna.rt.PathSolver`.
   - Executes a parameter sweep over multiple velocities ($v = 15, 30, 60 \text{ km/h}$) and feedback delays.
   - Computes NMSE and Achievable Sum-Rate Loss due to Doppler-induced phase shifts from mobility.
   - Embeds and renders inline Matplotlib plots showing NMSE (dB) versus Feedback Delay (ms) and Achievable Rate (bps/Hz) degradation.
4. Correctly executed the generated notebook using `jupyter nbconvert --execute` to run cells and bake plot graphics natively into the ipynb.

All objectives listed in `guidelines.md` have been fulfilled.
