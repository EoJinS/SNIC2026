# Guideline: GMTC Channel Reconstruction and NMSE Analysis

## 1. Objective
Extend the GMM-integrated notebook to implement the Gaussian-Mixture Transform Coding (GMTC) architecture. Compress the trajectory-based channel, reconstruct it, and evaluate the Normalized Mean Squared Error (NMSE).

## 2. References & Paths
- **Reference Paper**: "Fundamental Limits of CSI Compression.pdf"
- **Target Notebook**: `/home/ejseo/SNIC2026/experiments/covariance_visualization.ipynb` (Updated from Task 1)

## 3. Theoretical Framework: GMTC Architecture
Based on the reference paper, the GMTC compression and reconstruction must follow these exact steps:
1. **Component-Matched KLT**: Using the Eigendecomposition of the inferred state $\hat{c}$ ($R_{\hat{c}} = U_{\hat{c}} \Lambda_{\hat{c}} U_{\hat{c}}^H$), apply the Karhunen-Loève Transform:
   $$\tilde{h} = U_{\hat{c}}^H h$$
2. **Multi-Modal Reverse-Waterfilling (Compression)**: The ideal bit allocation is governed by a single global water level $\mu$. The allocated rate for mode $m$ is $r_{\hat{c},m}^* = \max(0, \log_2(\lambda_{\hat{c},m} / \mu))$. For simulation purposes, simulate the truncation/quantization by zeroing out transform coefficients where the eigenvalue is below the water level ($\lambda_{\hat{c},m} \le \mu$), or apply simulated scalar quantization matching $r_{\hat{c},m}^*$. Let the compressed coefficients be $\tilde{h}_q$.
3. **Inverse KLT (Reconstruction)**: Reconstruct the channel at the base station:
   $$\hat{h} = U_{\hat{c}} \tilde{h}_q$$

## 4. Implementation Tasks
1. **GMTC Pipeline**: Implement the component-matched KLT, truncation/quantization based on a defined water-level $\mu$, and inverse KLT for the CSI at each trajectory point.
2. **NMSE Calculation**: Calculate the NMSE between the original channel $h$ and the GMTC-reconstructed channel $\hat{h}$ at each physical location using:
   $$\text{NMSE (dB)} = 10 \log_{10} \left( \frac{\|h - \hat{h}\|_2^2}{\|h\|_2^2} \right)$$
3. **Visualization**: Create a line plot visualizing the NMSE degradation or variation as the user moves along the trajectory (NMSE vs. Trajectory Step).

## 5. Output Constraints
- Ensure the KLT operations strictly use the eigenvectors corresponding to the MAP GMM state ($\hat{c}$) identified at that specific trajectory step.
- Implement the reverse-waterfilling logic programmatically based on a tunable $\mu$ parameter.