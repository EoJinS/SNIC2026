# GMM_mobility.ipynb Refactoring Guideline

## 0. Objective

`GMM_mobility.ipynb` currently follows a structure where **the GMM is trained directly on the 50 trajectory points themselves, and then those same 50 points are classified against that trained model**. This causes two fundamental problems:

1. **Insufficient samples**: Fitting a 128-dimensional (or 2048-dimensional, if using 32×32 antennas) full covariance matrix with 50 or fewer samples (only 15–25 per cluster once split into K clusters) produces a rank-deficient covariance estimate.
2. **Mismatch with the research goal**: What we actually want to know is "which cluster of a **fixed, general spatial channel map (EM map)** does the mobile user currently belong to at their location" — not "how the 50 trajectory points cluster relative to each other."

**Direction**: Completely separate GMM training (map building) from trajectory querying (inference) using a **Two-Stage architecture**.

```
Stage A (run once, results saved to disk)
  Dense spatial grid (N_grid >> 50) -> GMM-EM training -> save cluster dictionary (U_c, lam_c, mu_c, pi_c)

Stage B (run repeatedly per trajectory, no training)
  Mobile user's trajectory h_tensor -> query the Stage A dictionary via predict_modes() only
  -> obtain per-timestep cluster label + the covariance R_c of that cluster
```

This document lists the **parts that are not yet implemented, or that still need validation**, as concrete tasks, based on the code already reflected in `GMM_mobility.ipynb`. antigravity-cli should process the items below in order.

---

## 1. Prerequisites (must be checked first)

- [ ] Open `GMM_mobility.ipynb` and inspect the current cell structure. Check whether the Stage A / Stage B cells described under "Target Architecture" already exist, and if so, under what assumptions they were written.
- [ ] Open `scene_builder.py` and `scene_engine.py` and check the signature of `setup_scene()`. Confirm whether it is currently hardcoded for the linear 50-point trajectory only.
- [ ] Confirm that `gmtc_em.py` and `run_em.py` are already included in the project and importable from the notebook (either same folder or on `sys.path`).
- [ ] Check the scene's coordinate range (coverage radius relative to the BS position) so the Stage A grid range (`x_range`, `y_range`) can be set to match the actual scene, rather than using an arbitrary ±50m.

---

## 2. Task 1 — Generalize `scene_engine.py` (top priority)

**Problem**: Currently `setup_scene(config)` only generates the fixed linear trajectory (50 points, 1m spacing) defined in `config.py`. For Stage A (grid) and Stage B (trajectory) to share this function, it must be able to accept an arbitrary array of positions.

**Tasks**:
- [ ] Add a new function `setup_scene_with_positions(config, positions: np.ndarray)` to `scene_engine.py`.
  - `positions`: an array of shape `(N, 3)`, whether it represents a grid or a trajectory.
  - Reuse the internal logic of the existing `setup_scene()` (BS placement, UE placement, batched Receiver creation), but replace only the trajectory-generation part with the `positions` argument.
  - Keep the existing `setup_scene(config)` for backward compatibility, refactoring it internally to call `setup_scene_with_positions(config, config.trajectory_positions)`.
- [ ] Verify that `compute_batch_paths` and `get_ofdm_channel` in `channel_engine.py` / `channel_analyzer.py` work correctly regardless of the number of positions (N). If `N_pos=50` is hardcoded anywhere internally, remove it and handle it dynamically.

**Validation**: Confirm that calling `setup_scene_with_positions(config, grid_positions)` and `setup_scene_with_positions(config, trajectory_positions)` each return an `h_tensor` with shape `(N, N_sub, N_tx)` that scales correctly with N.

---

## 3. Task 2 — Stage A: Build the EM Dictionary from a Spatial Grid

**Tasks**:
- [ ] Query the scene's actual coverage radius, then decide the grid resolution (initial value: around 30×30 to 40×40, adjust based on runtime).
- [ ] If ray tracing runtime becomes too long, apply one of the two alternatives below. antigravity-cli should check whether both alternatives are feasible, and first check whether the `scene.coverage_map()` API is available in the installed Sionna version.
  - **Alternative 1 (recommended)**: Use Sionna's grid-native API such as `scene.coverage_map()`.
  - **Alternative 2**: Keep using `setup_scene_with_positions`, but reduce the grid size or lower `max_depth` (e.g., `max_depth=2`).
- [ ] After computing the grid channel `h_grid (N_grid, N_sub, N_tx)`, merge multiple subcarriers to increase the number of training samples (subsample subcarriers via `sub_stride` and reshape). **Goal: sample count (N_grid × number of subcarriers) should be much greater than the vector dimension (128).** At minimum, adjust the grid/subcarrier count so the sample count is at least 10x the dimension.
- [ ] Train the GMM using `gmm_em_fit_fullcov` from `gmtc_em.py`, and save the result (`RealGMMParams`) to `output/gmm_spatial_dictionary.pt` (via `torch.save`).
- [ ] Confirm that the training log (`ll`, `dLL`) actually converges; if not, increase `num_iter` or adjust `eps`.
- [ ] Sweep K (number of clusters) over several values (e.g., 3, 4, 8), generate a `run_em.py`-style EM map (`Cell A4`) for each, and add a cell that visually compares which K best reflects the scene's physical structure (e.g., LOS/NLOS boundaries).

**Validation**: Add a cell that reloads the saved `gmm_spatial_dictionary.pt` and asserts that `params_map.U[c]` is indeed an orthonormal matrix (`U^T U ≈ I`) and that `params_map.lam[c]` is all positive.

---

## 4. Task 3 — Convert Trajectory Handling to Query-Only (Stage B)

**Tasks**:
- [ ] Confirm that the trajectory-related cells in the notebook **do not retrain the GMM**, and instead load `gmm_spatial_dictionary.pt` saved from Task 2 via `torch.load` and only call `predict_modes()`. Remove any remaining retraining code if it still exists.
- [ ] For labeling each trajectory timestep (t), use **the same subcarrier subsampling scheme as Task 2** (not just the single center subcarrier), and unify the approach using the "majority vote" method proposed earlier.
- [ ] Confirm that the "current covariance" $R_c$ at each timestep is not re-estimated from the trajectory data, but is instead reconstructed directly (via `reconstruct_complex_cov`) from the `U_c, lam_c` of the Stage A dictionary.

**Validation**: Print the count of each state among `c_hat_traj` for the 50 trajectory points, and confirm there is no state that appears only once or twice and is therefore statistically unstable (this should not be a problem anymore since Stage A now has plenty of grid samples).

---

## 5. Task 4 — Complete EM Map Visualization

**Tasks**:
- [ ] Follow `run_em.py`'s visualization style (legend, BS marker, `tab20`/`tab10` colormap, legend placement via `bbox_to_anchor`) to complete two types of plots in the notebook.
  1. **Pure spatial EM map** (Stage A4): grid points colored by cluster.
  2. **Trajectory-overlaid EM map** (Stage B2): the map above, overlaid with the trajectory path (black line) and per-timestep labels (solid dots).
- [ ] Include code to save both plots as PNGs to the `output/` folder (`plt.savefig(..., dpi=120)`), keeping filename conventions consistent with `run_em.py` (`em_map_{tag}_K{K}.png`).
- [ ] (Optional) Add a cell that arranges maps for multiple K values side by side as subplots in a single figure for comparison.

---

## 6. Task 5 — Re-validate the Connection with the NMSE / GMTC Compression Pipeline

Since separating Stage A/B changes the input to the existing Cell 7 (GMTC compression, NMSE evaluation), re-validation is needed.

- [ ] Update the references in the existing code so that `U_c_complex_list` and `lam_c_complex_list` come from **the Stage A dictionary**, not from a model trained on the trajectory itself.
- [ ] After recomputing NMSE, compare it against the previous run's results (average -14.32 dB, spikes at certain trajectory points) and record whether:
  - the spikes decrease because the dictionary is now well estimated, or
  - performance changes because the grid training data does not sufficiently cover the area around the actual trajectory path.
- [ ] If NMSE worsens, this indicates the Stage A grid did not sample the vicinity of the trajectory path densely enough. Consider increasing grid resolution specifically near the trajectory path (e.g., weighted sampling around the path instead of a uniform grid).

---

## 7. Deliverables Checklist (Definition of Done)

- [ ] `setup_scene_with_positions()` is added to `scene_engine.py` and reused by both the grid and the trajectory.
- [ ] The file `output/gmm_spatial_dictionary.pt` is created, and reloading it confirms valid parameters (orthonormal / positive eigenvalues).
- [ ] All GMM retraining code is completely removed from Stage B cells, leaving only `predict_modes()` calls.
- [ ] `output/em_map_spatial_K{K}.png` (pure spatial map) and `output/em_map_with_trajectory_K{K}.png` (trajectory overlay) are each generated.
- [ ] A summary comparing the re-evaluated NMSE results against the previous results (-14.32 dB, etc.) is recorded in the final notebook cell or a markdown cell.
- [ ] The entire notebook runs end-to-end without errors via `Run All Cells`.

---

## 8. Notes During Implementation

- Clearly state in code comments and markdown cells that Stage A and Stage B are **datasets with different purposes**, so future readers don't get confused again.
- Do **not** merge the ASU dataset (`asu_fd_32x32_stride32.npy`, `asu_pos.npz`) into this pipeline — the antenna count (32 vs 64) and the vectorization scheme (arbitrary chunking vs physical real∥imag) differ, making direct reuse impossible. If needed at all, keep it in a fully separate section only.
- Increasing grid resolution may cause GPU out-of-memory (OOM) errors, so re-confirm that `tf.config.experimental.set_memory_growth` is applied before running Stage A as well.
- Add a docstring or markdown explanation to every new function/cell so that the Stage A/B distinction is clear just from reading the notebook.