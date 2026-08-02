Target file : /home/ejseo/SNIC2026/experiments/mobility_simualtion_cir_ani.ipynb
**Update the plotting section of the script. Currently, the code generates a figure with three subplots: Original CFR, Average PDP (Linear), and Average PDP (dB). Please modify the third subplot based on the following instructions:**

**1. Remove the Average PDP (dB) plot:**
Remove the code that calculates and plots the Average Power Delay Profile in dB in the third axis (`axes[2]`).

**2. Replace it with a 'Phase vs. Delay' plot:**
In the third axis (`axes[2]`), create a stem plot that shows the phase of each path with respect to its delay.

**3. Phase Calculation:**
Calculate the phase of the complex channel impulse response (CIR) for each valid path. You can extract the phase from the time-averaged complex CIR using `np.angle(np.mean(cir_target_a, axis=1))` and convert the result from radians to degrees using `np.degrees()`.

**4. Plot Formatting:**

* Use `cir_target_tau_us` for the x-axis.
* Use the calculated phase (in degrees) for the y-axis.
* Set the title to `'Average Phase vs. Delay'`.
* Set the x-label to `'Delay ($\\mu s$)'` and the y-label to `'Phase (Degrees)'`.
* Keep the same stem plot styling (red markers, black stems) and grid lines as the previous subplots.