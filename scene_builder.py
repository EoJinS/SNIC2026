import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver

def build_scene(config, frequency):
    scene = load_scene(sionna.rt.scene.etoile)
    scene.frequency = frequency
    
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
        pattern="tr38901",
        polarization="V"
    )
    
    bs = Transmitter("bs", position=[0, 0, 25])
    ue = Receiver("ue", position=[50, 10, 1.5])
    
    if "bs" in scene.transmitters:
        scene.remove("bs")
    if "ue" in scene.receivers:
        scene.remove("ue")
        
    scene.add(bs)
    scene.add(ue)
    return scene, bs, ue
