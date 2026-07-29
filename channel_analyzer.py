import tensorflow as tf
from sionna.rt import PathSolver
from sionna.rt.utils import subcarrier_frequencies

def compute_batch_paths(scene, max_depth=3):
    solver = PathSolver()
    paths = solver(scene=scene, max_depth=max_depth)
    return paths

def get_ofdm_channel(paths, config):
    freqs = subcarrier_frequencies(config.fft_size, config.scs)
    
    cfr = paths.cfr(
        frequencies=freqs,
        sampling_frequency=config.scs,
        num_time_steps=1,
        normalize_delays=False,
        normalize=True,
        out_type="tf"
    )
    
    # cfr shape: [num_rx=N_pos, num_rx_ant=1, num_tx=1, num_tx_ant=64, num_time_steps=1, num_subcarriers]
    # Extract all receivers, active subcarriers, and time_step 0
    h_freq = cfr[..., 0, :config.num_active_subcarriers]
    
    # Reshape to [N_pos, N_tx_ant, active_subcarriers]
    num_rx = tf.shape(h_freq)[0]
    h_reshaped = tf.reshape(h_freq, [num_rx, config.num_bs_ant_rows*config.num_bs_ant_cols, config.num_active_subcarriers])
    
    # Transpose to [N_pos, N_sub, N_tx]
    h_transposed = tf.transpose(h_reshaped, [0, 2, 1])
    return h_transposed

def compute_spatial_covariance(h_tensor):
    # h_tensor: [N_pos, N_sub, N_tx]
    # expand to [N_pos, N_sub, N_tx, 1]
    h_exp = tf.expand_dims(h_tensor, axis=-1)
    
    # Outer product for each position and subcarrier: h * h^H -> [N_pos, N_sub, N_tx, N_tx]
    R_per_pos = tf.matmul(h_exp, h_exp, adjoint_b=True)
    
    # Average across trajectory (N_pos) -> [N_sub, N_tx, N_tx]
    R_mean = tf.reduce_mean(R_per_pos, axis=0)
    
    return R_mean
