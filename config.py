from dataclasses import dataclass

@dataclass
class Config:
    fc: float = 3.5e9
    scs: float = 30e3
    fft_size: int = 1024
    num_active_subcarriers: int = 600
    
    num_bs_ant_rows: int = 8
    num_bs_ant_cols: int = 8
    num_ue_ant_rows: int = 1
    num_ue_ant_cols: int = 1
    
    num_pos: int = 50
    pos_start: tuple = (50.0, 10.0, 1.5)
    pos_step: tuple = (0.0, 1.0, 0.0) # 1 meter spaced
