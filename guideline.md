# Experiment Guide: GMM-based MMSE Channel Estimation

## Target File
`/home/ejseo/SNIC2026/GMM_static_POSTECH.ipynb`

## Overview
Update the static spatial channel dictionary simulation to evaluate MMSE channel estimation performance using a Gaussian Mixture Model (GMM) trained via the EM algorithm. Apply new OFDM numerology, inject noise, and evaluate the Normalized Mean Square Error (NMSE) across different GMM component sizes.

## Task 1: Validation of Current Results
- Briefly analyze the generated CIR and CFR graphs to confirm they reflect correct multipath fading characteristics for a static grid with `normalize_delays=False`.

## Task 2: Update Simulation Settings
Modify the `Config` class and scene setup to match the following constraints:
- **Number of Antennas**: Update the BS PlanarArray to 32 elements (e.g., `num_bs_ant_rows = 8`, `num_bs_ant_cols = 4`).
- **Number of Subcarriers**: Set `num_active_subcarriers = 32`. Adjust the `fft_size` accordingly (e.g., to 64) to maintain simulation efficiency.

## Task 3: Signal Transmission & Noise (SNR = 10 dB)
- **Pilots**: Construct an OFDM resource grid that transmits pilots across *all* OFDM symbols (dense pilot allocation).
- **Noise Injection**: Add AWGN to the received signal to accurately reflect a Tx SNR of 10 dB.

## Task 4: Channel Count & Additional Analysis
- Add code to print the exact number of channels generated and used for training (e.g., Total Paths / Valid Grid Points x Subcarriers x Rx/Tx Antennas).
- Write a brief analysis block to extract and print spatial correlation characteristics (e.g., the condition number of the spatial covariance matrix or eigenvalue spread) to verify spatial sparsity.

## Task 5: GMM Training & MMSE Estimation Loop
- Set up a loop to evaluate the GMM for different numbers of components: K = 4, 8, and 16.
- For each K:
  1. Train the GMM using the provided `gmm_em_fit_fullcov` (EM algorithm).
  2. Perform MMSE channel estimation utilizing the learned GMM spatial priors.
  3. Compute the Normalized Mean Square Error (NMSE) of the estimated channel versus the true channel.

## Task 6: NMSE Visualization
- Add a new cell to plot the final NMSE results. 
- The X-axis must represent the GMM components (K = 4, 8, 16) and the Y-axis must represent the NMSE (in dB scale). 
- Ensure the plot includes gridlines, appropriate labels, and a legend.