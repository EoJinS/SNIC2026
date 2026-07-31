import tensorflow as tf
import sionna
import numpy as np

scene = sionna.rt.load_scene(sionna.rt.scene.etoile)
scene.tx_array = sionna.rt.PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")
scene.rx_array = sionna.rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="dipole", polarization="V")
tx = sionna.rt.Transmitter(name="tx", position=[0, 0, 30])
scene.add(tx)
rx = sionna.rt.Receiver(name="rx", position=[10, 10, 1.5])
scene.add(rx)

solver = sionna.rt.PathSolver()
paths = solver(scene, max_depth=1)

a_real, a_imag = paths.a
a_real_np = np.array(a_real)
a_imag_np = np.array(a_imag)

power = a_real_np**2 + a_imag_np**2
# Sum over rx_ant (axis 1) and tx_ant (axis 3)
path_power = np.sum(power, axis=(1, 3))
print("path_power shape:", path_power.shape) # Should be [num_rx, num_tx, num_paths]

vertices_np = np.array(paths.vertices)
print("vertices shape:", vertices_np.shape) # Should be [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, 3] or [max_depth, num_rx, num_tx, num_paths, 3]
