from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class SimulationConfig:
    fc_tdd: float = 3.5e9
    fc_fdd_ul: float = 1.9e9
    fc_fdd_dl: float = 2.1e9
    scs: float = 30e3
    fft_size: int = 1024
    num_active_subcarriers: int = 600
    
    num_bs_ant_rows: int = 8
    num_bs_ant_cols: int = 8
    num_ue_ant_rows: int = 1
    num_ue_ant_cols: int = 1
    
    velocities: List[float] = field(default_factory=lambda: [15.0, 30.0, 60.0]) # km/h
    delta_t_list: List[float] = field(default_factory=lambda: [0, 1e-3, 2e-3, 5e-3, 10e-3])
