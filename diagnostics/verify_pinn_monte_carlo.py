import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.uav_comm.components.antenna import array_locs, pert2d_null_multi, phase_code_finder, find_gain_of_tphi
from src.uav_comm.components.pinn_surrogate import PairwiseInterferencePINN

def run_monte_carlo(N_trials=1000, bits=1):
    print(f"\n--- Running Monte Carlo PINN Validation ({bits}-bit Quantization, {N_trials} Trials) ---")

    D, PhaseTable = array_locs(8)
    hpbw = 20.0
    pinn = PairwiseInterferencePINN(hpbw_deg=hpbw, quantization_bits=bits)

    noise_db = -90
    bandwidth = 0.35
    max_range = 500.0
    uav_height = 50.0

    true_scores = []
    pinn_scores = []

    np.random.seed(42)

    # We sweep with a progress bar since the physics engine is heavy
    for _ in tqdm(range(N_trials), desc="Simulating Random Geometries"):
        # 1. Randomize UAV Location
        uav_pos = np.random.uniform(-max_range/2, max_range/2, size=2)

        # 2. Randomize User Locations
        u1_pos = np.random.uniform(-max_range, max_range, size=2)
        u2_pos = np.random.uniform(-max_range, max_range, size=2)

        # 3. Calculate True Geometry (Angles and Distances)
        # Target User 1
        d1 = u1_pos - uav_pos
        dist_3d_1 = np.sqrt(d1[0]**2 + d1[1]**2 + uav_height**2)
        theta_1 = np.degrees(np.arccos(uav_height / dist_3d_1))
        phi_1 = np.degrees(np.arctan2(d1[1], d1[0]))
        R1 = np.linalg.norm(d1)

        # Target User 2 (Interferer)
        d2 = u2_pos - uav_pos
        dist_3d_2 = np.sqrt(d2[0]**2 + d2[1]**2 + uav_height**2)
        theta_2 = np.degrees(np.arccos(uav_height / dist_3d_2))
        phi_2 = np.degrees(np.arctan2(d2[1], d2[0]))
        R2 = np.linalg.norm(d2)

        # 4. Calculate Baseline Capacity (User 1 alone)
        ph_baseline = phase_code_finder(D, PhaseTable, theta_1, phi_1, quantization_bits=bits)
        gain_baseline_db = find_gain_of_tphi(theta_1, phi_1, ph_baseline, D)
        sig_lin_base = 10 ** (gain_baseline_db / 10)
        noise_lin = 10 ** (noise_db / 10)
        sinr_base = sig_lin_base / noise_lin
        cap_base = bandwidth * np.log2(1 + sinr_base)

        # Avoid zero/low capacity baseline division errors
        if cap_base < 1e-6:
            continue

        # 5. Physics Engine: Serve U1 while nulling U2
        sig_db, int_db = pert2d_null_multi(D, PhaseTable, theta_1, phi_1, R1, np.array([theta_2]), np.array([phi_2]), np.array([R2]), noise_db, quantization_bits=bits)

        sig_lin = 10 ** (sig_db / 10)
        int_lin = 10 ** (int_db[0] / 10)
        sinr = sig_lin / (int_lin + noise_lin)
        cap = bandwidth * np.log2(1 + sinr)

        retention = cap / cap_base
        true_scores.append(retention)

        # 6. PINN Score
        delta_phi = phi_1 - phi_2
        dist_ratio = max(R1, R2) / (min(R1, R2) + 1e-6)

        dp_tensor = torch.tensor([delta_phi], dtype=torch.float32)
        dist_tensor = torch.tensor([dist_ratio], dtype=torch.float32)

        p_score = pinn(dp_tensor, dist_tensor).item()
        pinn_scores.append(p_score)

    true_scores = np.array(true_scores)
    pinn_scores = np.array(pinn_scores)

    mae = np.mean(np.abs(true_scores - pinn_scores))
    rmse = np.sqrt(np.mean((true_scores - pinn_scores)**2))
    corr = np.corrcoef(true_scores, pinn_scores)[0, 1]

    print("\n--- Results ---")
    print(f"Total Valid Trials: {len(true_scores)}")
    print(f"Mean True Retention: {np.mean(true_scores):.4f}")
    print(f"Mean PINN Predicted: {np.mean(pinn_scores):.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Err(RMSE): {rmse:.4f}")
    print(f"Pearson Correlation (R):   {corr:.4f}")

    return mae, rmse, corr

if __name__ == "__main__":
    run_monte_carlo(N_trials=1000, bits=1)
    run_monte_carlo(N_trials=1000, bits=2)
