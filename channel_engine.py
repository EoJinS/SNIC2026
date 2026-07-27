import tensorflow as tf
import sionna
from sionna.rt.utils import subcarrier_frequencies

from sionna.rt import PathSolver

def compute_paths(scene, max_depth=3):
    solver = PathSolver()
    paths = solver(scene=scene, max_depth=max_depth)
    paths.normalize_delays = False
    return paths

def apply_aging_and_get_ofdm_channel(paths, config, frequency, delta_t):
    freqs = subcarrier_frequencies(config.fft_size, config.scs)
    
    if delta_t <= 1e-12:
        dt_eff = 1.0 # arbitrary
        steps = 1
    else:
        dt_eff = delta_t
        steps = 2
        
    cfr = paths.cfr(
        frequencies=freqs,
        sampling_frequency=1.0/dt_eff,
        num_time_steps=steps,
        normalize_delays=False,
        normalize=True,
        out_type="tf"
    )
    
    if delta_t <= 1e-12:
        h_freq = cfr[..., 0, :]
    else:
        h_freq = cfr[..., 1, :]
        
    h_freq_active = h_freq[..., :config.num_active_subcarriers]
    return h_freq_active
