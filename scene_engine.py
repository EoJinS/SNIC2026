import tensorflow as tf
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver

def setup_scene(config):
    scene = load_scene(sionna.rt.scene.etoile)
    scene.frequency = config.fc
    
    scene.tx_array = PlanarArray(
        num_rows=config.num_bs_ant_rows,
        num_cols=config.num_bs_ant_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V"
    )
    
    scene.rx_array = PlanarArray(
        num_rows=config.num_ue_ant_rows,
        num_cols=config.num_ue_ant_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="V"
    )
    
    bs = Transmitter("bs", position=[0, 0, 25])
    
    if "bs" in scene.transmitters:
        scene.remove("bs")
    scene.add(bs)
    
    # Remove existing receivers
    for name in list(scene.receivers.keys()):
        scene.remove(name)
        
    positions = []
    # Add multiple receivers for trajectory batching
    for i in range(config.num_pos):
        pos = [config.pos_start[0] + i*config.pos_step[0],
               config.pos_start[1] + i*config.pos_step[1],
               config.pos_start[2] + i*config.pos_step[2]]
        ue = Receiver(f"ue_{i}", position=pos)
        ue.velocity = [0.0, 15.0, 0.0]
        scene.add(ue)
        positions.append(pos)
        
    pos_tensor = tf.cast(tf.stack(positions, axis=0), tf.float32) # [N_pos, 3]
    return scene, bs, None, pos_tensor
