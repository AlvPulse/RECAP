import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from src.uav_comm.components.antenna import array_locs, pert2d_null_multi, phase_code_finder, find_gain_of_tphi
from src.uav_comm.components.pinn_surrogate import PairwiseInterferencePINN

def run_accuracy_test(bits=1, N_samples=200):
    print(f"\n--- Running PINN Accuracy Test ({bits}-bit Quantization) ---")

    # Initialize Physics and PINN
    D, PhaseTable = array_locs(8) # 8x8 array -> HPBW ~ 20 deg
    hpbw = 20.0
    pinn = PairwiseInterferencePINN(hpbw_deg=hpbw, quantization_bits=bits)

    # Test Parameters
    theta_t = 30
    R = 100
    RN = np.array([100])
    noise_db = -90
    bandwidth = 0.35

    # Fixed Target User at phi=0
    phi_t = 0

    angles_to_test = np.linspace(0, 359, N_samples)
    true_scores = []
    pinn_scores = []

    # 1. Calculate Baseline Capacity (Serving Target Alone)
    ph_baseline = phase_code_finder(D, PhaseTable, theta_t, phi_t, quantization_bits=bits)
    gain_baseline_db = find_gain_of_tphi(theta_t, phi_t, ph_baseline, D)
    sig_lin_base = 10 ** (gain_baseline_db / 10)
    noise_lin = 10 ** (noise_db / 10)
    sinr_base = sig_lin_base / noise_lin
    cap_base = bandwidth * np.log2(1 + sinr_base)

    print("Sweeping angular separations...")
    for phi_int in angles_to_test:
        # Avoid exact 0 separation math errors in physics engine if any
        if abs(phi_int) < 1.0 or abs(phi_int - 360.0) < 1.0:
            true_scores.append(0.0)

            dp = torch.tensor([phi_int], dtype=torch.float32)
            dist_ratio = torch.tensor([1.0], dtype=torch.float32)
            p_score = pinn(dp, dist_ratio).item()
            pinn_scores.append(p_score)
            continue

        theta_n = np.array([30])
        phi_n = np.array([phi_int])

        # 2. Physics Engine: Serve Target while Nulling Interferer
        sig_db, int_db = pert2d_null_multi(D, PhaseTable, theta_t, phi_t, R, theta_n, phi_n, RN, noise_db, quantization_bits=bits)

        # Calculate resulting capacity
        sig_lin = 10 ** (sig_db / 10)
        int_lin = 10 ** (int_db[0] / 10)
        sinr = sig_lin / (int_lin + noise_lin)
        cap = bandwidth * np.log2(1 + sinr)

        # True Score is retention ratio
        retention = cap / cap_base
        true_scores.append(retention)

        # 3. PINN Score
        dp = torch.tensor([phi_int], dtype=torch.float32)
        dist_ratio = torch.tensor([1.0], dtype=torch.float32) # Same distance
        p_score = pinn(dp, dist_ratio).item()
        pinn_scores.append(p_score)

    true_scores = np.array(true_scores)
    pinn_scores = np.array(pinn_scores)

    # Calculate Error Metrics
    mae = np.mean(np.abs(true_scores - pinn_scores))
    corr = np.corrcoef(true_scores, pinn_scores)[0, 1]

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Pearson Correlation:       {corr:.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(angles_to_test, true_scores, label='Physics Simulation (True Retention)', color='blue')
    plt.plot(angles_to_test, pinn_scores, label='PINN Surrogate Prediction', color='orange', linestyle='--')
    plt.title(f'PINN Accuracy vs Physics ({bits}-bit Quantization)\nMAE: {mae:.4f} | Corr: {corr:.4f}')
    plt.xlabel('Angular Separation (Degrees)')
    plt.ylabel('Capacity Retention Score [0, 1]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'diagnostics/pinn_accuracy_{bits}bit.png')
    plt.close()
    print(f"Plot saved to diagnostics/pinn_accuracy_{bits}bit.png")

if __name__ == "__main__":
    run_accuracy_test(bits=1)
    run_accuracy_test(bits=2)
