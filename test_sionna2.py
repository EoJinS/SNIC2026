import tensorflow as tf
import sionna
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

scene = sionna.rt.load_scene(sionna.rt.scene.etoile)
scene.tx_array = sionna.rt.PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")
scene.rx_array = sionna.rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="dipole", polarization="V")

tx = sionna.rt.Transmitter(name="tx", position=[0, 0, 30])
scene.add(tx)
rx = sionna.rt.Receiver(name="rx", position=[10, 10, 1.5])
scene.add(rx)

solver = sionna.rt.PathSolver()
paths = solver(scene, max_depth=1)

print("type of paths.a:", type(paths.a))
print("paths.a[0].shape:", paths.a[0].shape)
a_real, a_imag = paths.a
power = a_real**2 + a_imag**2
print("power shape:", power.shape)
