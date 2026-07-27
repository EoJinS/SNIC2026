"""
Sionna RT 기반 Massive MIMO-OFDM Mobility 시뮬레이터
TDD/FDD 환경에서 CSI Aging 및 Channel Mismatch 분석
"""

import numpy as np
import matplotlib.pyplot as plt
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
from sionna.rt.utils import subcarrier_frequencies


class MassiveMIMOOFDMMobilitySimulator:
    """
    Sionna RT를 이용한 Massive MIMO-OFDM 채널 시뮬레이터
    """

    def __init__(self,
                 carrier_freq=3.5e9,        # Hz
                 num_subcarriers=1024,
                 subcarrier_spacing=30e3,   # 30 kHz (5G NR)
                 num_ofdm_symbols=14,       # 1 slot
                 num_bs_ant_rows=8,         # 64 antennas
                 num_bs_ant_cols=8,
                 num_ue_ant_rows=1,
                 num_ue_ant_cols=1,
                 scene_name=sionna.rt.scene.simple_street_canyon_with_cars):
        
        # --- OFDM / System Params ---
        self.carrier_freq = carrier_freq
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing = subcarrier_spacing
        self.num_ofdm_symbols = num_ofdm_symbols
        self.num_bs_ant = num_bs_ant_rows * num_bs_ant_cols
        self.num_ue_ant = num_ue_ant_rows * num_ue_ant_cols
        self.ofdm_symbol_duration = 1.0 / subcarrier_spacing  # ~33.3 us
        
        # --- Load Scene ---
        self.scene = load_scene(scene_name, merge_shapes=False)
        self.scene.frequency = carrier_freq
        
        # --- Antenna Arrays ---
        # BS: Massive MIMO UPA
        self.scene.tx_array = PlanarArray(
            num_rows=num_bs_ant_rows,
            num_cols=num_bs_ant_cols,
            vertical_spacing=0.5,      # lambda/2
            horizontal_spacing=0.5,
            pattern="tr38901",         # 3GPP TR 38.901
            polarization="V"
        )
        # UE: Small array
        self.scene.rx_array = PlanarArray(
            num_rows=num_ue_ant_rows,
            num_cols=num_ue_ant_cols,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V"
        )
        
        # --- Subcarrier Frequencies ---
        self.frequencies = subcarrier_frequencies(
            num_subcarriers, subcarrier_spacing
        )
        
        # --- Path Solver ---
        self.path_solver = PathSolver()
        self.bs = None
        self.ue = None

    # ============================================================
    # 1. Deployment & Mobility
    # ============================================================
    def deploy_bs_ue(self, bs_pos, ue_pos, bs_ori=(0,0,0), ue_ori=(0,0,0)):
        """BS와 UE를 장면에 배치"""
        self.bs = Transmitter("bs", position=bs_pos, orientation=bs_ori)
        self.ue = Receiver("ue", position=ue_pos, orientation=ue_ori)
        self.scene.add(self.bs)
        self.scene.add(self.ue)

    def set_ue_mobility(self, velocity, moving_objects=None):
        """
        UE 및 장면 내 이동체 속도 설정
        velocity: [vx, vy, vz] in m/s
        moving_objects: {"car_1": [vx,vy,vz], ...}
        """
        self.scene.get("ue").velocity = velocity
        if moving_objects:
            for name, vel in moving_objects.items():
                if name in self.scene.objects:
                    self.scene.get(name).velocity = vel

    # ============================================================
    # 2. Channel Computation (Doppler-based time evolution)
    # ============================================================
    def compute_channel(self, max_depth=3, num_time_steps=None,
                        refraction=False, diffraction=False):
        """
        Sionna RT로 CFR 계산. Doppler shift를 이용한 시간 진화 포함.
        
        Returns
        -------
        cfr : np.ndarray
            shape [1, 1, N_ue_ant, 1, N_bs_ant, T, F]
        paths : sionna.rt.Paths
        """
        if num_time_steps is None:
            num_time_steps = self.num_ofdm_symbols

        paths = self.path_solver(
            scene=self.scene,
            max_depth=max_depth,
            refraction=refraction,
            diffraction=diffraction
        )

        # Doppler 기반 시간 진화 (짧은 간격에 매우 정확)
        cfr = paths.cfr(
            frequencies=self.frequencies,
            sampling_frequency=self.subcarrier_spacing,  # 1/T_s
            num_time_steps=num_time_steps,
            normalize_delays=False,
            normalize=True,
            out_type="numpy"
        )
        return cfr, paths

    def extract_mimo_matrix(self, cfr, time_idx, freq_idx=None):
        """
        Sionna CFR 텐서에서 MIMO 채널 행렬 추출
        cfr: [1, 1, N_ue, 1, N_bs, T, F]
        
        Returns
        -------
        H : np.ndarray
            [N_ue, N_bs] if freq_idx given, else [N_ue, N_bs, F]
        """
        h = np.squeeze(cfr, axis=(0, 1, 3))  # [N_ue, N_bs, T, F]
        if freq_idx is not None:
            return h[:, :, time_idx, freq_idx]
        return h[:, :, time_idx, :]

    # ============================================================
    # 3. TDD CSI Aging Analysis
    # ============================================================
    def analyze_tdd_csi_aging(self,
                                pilot_symbol=0,
                                data_symbol=7,
                                snr_db=20,
                                max_depth=3):
        """
        TDD: UL Pilot (t=pilot_symbol)에서 추정한 CSI를 
        DL Data (t=data_symbol)에 사용할 때의 aging 분석.
        
        Assumption: TDD reciprocity (H_DL ≈ H_UL^T)
        """
        # 1슬롯 내 시간 진화 계산
        cfr, paths = self.compute_channel(
            max_depth=max_depth,
            num_time_steps=self.num_ofdm_symbols
        )
        
        # H shape: [N_ue, N_bs, T, F]
        H = np.squeeze(cfr, axis=(0, 1, 3))
        
        H_est = H[:, :, pilot_symbol, :]      # 추정 CSI
        H_actual = H[:, :, data_symbol, :]    # 실제 채널
        
        nmse = np.zeros(self.num_subcarriers)
        correlation = np.zeros(self.num_subcarriers)
        sinr_perfect = np.zeros(self.num_subcarriers)
        sinr_aged = np.zeros(self.num_subcarriers)
        
        for f in range(self.num_subcarriers):
            Hf_est = H_est[:, :, f]
            Hf_act = H_actual[:, :, f]
            
            h_est = Hf_est.flatten()
            h_act = Hf_act.flatten()
            
            # NMSE
            nmse[f] = np.mean(np.abs(h_act - h_est)**2) / np.mean(np.abs(h_act)**2)
            
            # Temporal correlation
            correlation[f] = (np.abs(np.vdot(h_act, h_est)) / 
                              (np.linalg.norm(h_act) * np.linalg.norm(h_est)))
            
            # Single-stream MRT precoding analysis (N_ue = 1 가정)
            if self.num_ue_ant == 1:
                w = Hf_est.conj().T           # [N_bs, 1]
                w = w / (np.linalg.norm(w) + 1e-12)
                
                snr_lin = 10**(snr_db / 10)
                sinr_perfect[f] = np.abs(Hf_act @ (Hf_act.conj().T / 
                                        (np.linalg.norm(Hf_act) + 1e-12)))**2 * snr_lin
                sinr_aged[f] = np.abs(Hf_act @ w)**2 * snr_lin
        
        aging_time = (data_symbol - pilot_symbol) * self.ofdm_symbol_duration
        
        return {
            'nmse': nmse,
            'correlation': correlation,
            'sinr_perfect': sinr_perfect,
            'sinr_aged': sinr_aged,
            'aging_time': aging_time,
            'H_est': H_est,
            'H_actual': H_actual,
            'paths': paths
        }

    # ============================================================
    # 4. FDD CSI Mismatch Analysis
    # ============================================================
    def analyze_fdd_csi_mismatch(self,
                                   dl_freq=3.5e9,
                                   ul_freq=2.1e9,
                                   feedback_delay_symbols=14,
                                   max_depth=3):
        """
        FDD: DL 주파수에서 추정된 outdated CSI가 
        UL 주파수 + feedback delay 후에 사용될 때의 mismatch 분석.
        """
        # --- Step 1: DL Channel (outdated) ---
        self.scene.frequency = dl_freq
        self.frequencies = subcarrier_frequencies(
            self.num_subcarriers, self.subcarrier_spacing
        )
        
        cfr_dl, _ = self.compute_channel(
            max_depth=max_depth,
            num_time_steps=feedback_delay_symbols + 1
        )
        H_dl = np.squeeze(cfr_dl, axis=(0, 1, 3))
        H_dl_outdated = H_dl[:, :, 0, :]  # t=0 (feedback 시점)
        
        # --- Step 2: UL Channel (actual current) ---
        self.scene.frequency = ul_freq
        self.frequencies = subcarrier_frequencies(
            self.num_subcarriers, self.subcarrier_spacing
        )
        
        cfr_ul, _ = self.compute_channel(max_depth=max_depth, num_time_steps=1)
        H_ul = np.squeeze(cfr_ul, axis=(0, 1, 3))
        H_ul_actual = H_ul[:, :, 0, :]
        
        # --- Restore ---
        self.scene.frequency = self.carrier_freq
        self.frequencies = subcarrier_frequencies(
            self.num_subcarriers, self.subcarrier_spacing
        )
        
        # --- Metrics ---
        nmse = np.zeros(self.num_subcarriers)
        for f in range(self.num_subcarriers):
            h_dl = H_dl_outdated[:, :, f].flatten()
            h_ul = H_ul_actual[:, :, f].flatten()
            nmse[f] = np.mean(np.abs(h_ul - h_dl)**2) / np.mean(np.abs(h_ul)**2)
        
        return {
            'nmse': nmse,
            'H_dl_outdated': H_dl_outdated,
            'H_ul_actual': H_ul_actual,
            'feedback_delay': feedback_delay_symbols * self.ofdm_symbol_duration
        }

    # ============================================================
    # 5. Visualization
    # ============================================================
    def plot_tdd_aging(self, results):
        """TDD Aging 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # NMSE
        axes[0,0].plot(10*np.log10(results['nmse']))
        axes[0,0].set_title(f"TDD NMSE (Δt={results['aging_time']*1e6:.1f} μs)")
        axes[0,0].set_xlabel("Subcarrier"); axes[0,0].set_ylabel("NMSE [dB]")
        axes[0,0].grid(True)
        
        # Correlation
        axes[0,1].plot(results['correlation'])
        axes[0,1].set_title("Temporal Correlation")
        axes[0,1].set_xlabel("Subcarrier"); axes[0,1].set_ylabel("|ρ|")
        axes[0,1].grid(True)
        
        # Channel magnitude snapshot
        axes[1,0].plot(np.abs(results['H_est'][0,0,:]), label='H(t=0)')
        axes[1,0].plot(np.abs(results['H_actual'][0,0,:]), label='H(t=Δt)')
        axes[1,0].set_title("|H(f)| Snapshot")
        axes[1,0].legend(); axes[1,0].grid(True)
        
        # Doppler distribution
        dop = results['paths'].doppler.numpy().flatten()
        axes[1,1].hist(dop, bins=50)
        axes[1,1].set_title("Doppler Shift Distribution")
        axes[1,1].set_xlabel("Hz"); axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()


# ============================================================
# Main Example
# ============================================================
if __name__ == "__main__":
    # 1. 초기화 (64x1 Massive MIMO, 1x1 UE)
    sim = MassiveMIMOOFDMMobilitySimulator(
        carrier_freq=3.5e9,
        num_subcarriers=1024,
        subcarrier_spacing=30e3,
        num_ofdm_symbols=14,
        num_bs_ant_rows=8,
        num_bs_ant_cols=8,
        num_ue_ant_rows=1,
        num_ue_ant_cols=1
    )
    
    # 2. BS/UE 배치
    sim.deploy_bs_ue(
        bs_pos=[10, -20, 25],
        ue_pos=[10, 5, 1.5]
    )
    
    # 3. 이동성 설정 (30 km/h ≈ 8.33 m/s, x축 방향)
    sim.set_ue_mobility(velocity=[8.33, 0, 0])
    
    # 4. TDD CSI Aging 분석
    #    Pilot: symbol 0, Data: symbol 7 (gap ≈ 233 μs)
    tdd_res = sim.analyze_tdd_csi_aging(
        pilot_symbol=0,
        data_symbol=7,
        snr_db=20,
        max_depth=3
    )
    print(f"[TDD] Aging Time: {tdd_res['aging_time']*1e6:.1f} μs")
    print(f"[TDD] Avg NMSE: {10*np.log10(np.mean(tdd_res['nmse'])):.2f} dB")
    print(f"[TDD] Avg Correlation: {np.mean(tdd_res['correlation']):.4f}")
    
    # 5. FDD CSI Mismatch 분석
    fdd_res = sim.analyze_fdd_csi_mismatch(
        dl_freq=3.5e9,
        ul_freq=2.1e9,
        feedback_delay_symbols=14,  # 1 slot delay
        max_depth=3
    )
    print(f"\n[FDD] Feedback Delay: {fdd_res['feedback_delay']*1e6:.1f} μs")
    print(f"[FDD] Avg NMSE: {10*np.log10(np.mean(fdd_res['nmse'])):.2f} dB")
    
    # 6. 시각화
    sim.plot_tdd_aging(tdd_res)