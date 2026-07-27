import tensorflow as tf

def zf_precoder(h_est):
    h_est_t = tf.transpose(h_est, [2, 0, 1])
    h_est_conj_t = tf.linalg.adjoint(h_est_t)
    h_h_conj_t = tf.matmul(h_est_t, h_est_conj_t)
    inv = tf.linalg.inv(h_h_conj_t + 1e-9 * tf.eye(1, dtype=h_h_conj_t.dtype))
    w_zf = tf.matmul(h_est_conj_t, inv)
    w_zf = w_zf / tf.cast(tf.norm(w_zf, axis=1, keepdims=True), tf.complex64)
    return w_zf
