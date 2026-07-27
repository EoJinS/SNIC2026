import tensorflow as tf

def calculate_nmse(h_true, h_est):
    diff = h_true - h_est
    mse = tf.reduce_mean(tf.abs(diff)**2)
    norm = tf.reduce_mean(tf.abs(h_true)**2)
    return float(mse / (norm + 1e-12))

def calculate_achievable_rate(h_true, w_zf, snr_db=20.0):
    h_true_t = tf.transpose(h_true, [2, 0, 1])
    eff_channel = tf.matmul(h_true_t, w_zf)
    signal_power = tf.abs(eff_channel)**2
    snr_lin = 10.0**(snr_db / 10.0)
    rate = tf.math.log(1.0 + snr_lin * tf.cast(signal_power, tf.float32)) / tf.math.log(2.0)
    return float(tf.reduce_mean(rate))
