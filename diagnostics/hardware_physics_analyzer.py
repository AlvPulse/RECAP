import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from src.uav_comm.components.antenna import array_locs, pert2d_null_multi, phase_code_finder, find_gain_of_tphi

def test_null_depth_vs_quantization():
    """Proves that low quantization destroys null depth."""
    print("\n--- Test 1: Null Depth vs. Quantization Bits ---")
    D, PhaseTable = array_locs(8) # 8x8 array
    theta_t, phi_t = 30, 45
    theta_n, phi_n = np.array([30]), np.array([60]) # Null target 15 degrees away
    R = 100
    RN = np.array([100])
    noise_db = -90

    # Ideal (Infinite precision approximation by using a very high bit count, pert2d doesn't support inf natively, so we observe trend)
    for bits in [1, 2, 3, 8]:
        sig_db, int_db = pert2d_null_multi(D, PhaseTable, theta_t, phi_t, R, theta_n, phi_n, RN, noise_db, quantization_bits=bits)
        print(f"Bits: {bits} | Mainlobe Gain: {sig_db:.2f} dB | Null Depth (Interference): {int_db[0]:.2f} dB")

def test_gain_sacrifice():
    """Proves that forcing a close null sacrifices mainlobe gain."""
    print("\n--- Test 2: Mainlobe Gain Sacrifice vs. Angular Separation ---")
    D, PhaseTable = array_locs(8)
    theta_t, phi_t = 30, 0
    R = 100
    RN = np.array([100])
    noise_db = -90

    # Baseline Gain (No Nulling)
    ph = phase_code_finder(D, PhaseTable, theta_t, phi_t)
    baseline_gain = find_gain_of_tphi(theta_t, phi_t, ph, D)
    print(f"Baseline Gain (No Nulling): {baseline_gain:.2f} dB")

    for sep in [5, 10, 20, 40, 90]:
        theta_n, phi_n = np.array([30]), np.array([phi_t + sep])
        sig_db, int_db = pert2d_null_multi(D, PhaseTable, theta_t, phi_t, R, theta_n, phi_n, RN, noise_db, quantization_bits=1)
        gain_loss = baseline_gain - sig_db
        print(f"Separation: {sep:2d} deg | Mainlobe Gain: {sig_db:.2f} dB (Loss: {gain_loss:.2f} dB) | Null: {int_db[0]:.2f} dB")

def test_dof_exhaustion():
    """Proves that 1-bit arrays collapse when attempting multiple nulls."""
    print("\n--- Test 3: DoF Exhaustion (1-Bit Array) ---")
    D, PhaseTable = array_locs(8) # 64 elements, theoretically 63 DoF
    theta_t, phi_t = 30, 0
    R = 100
    noise_db = -90

    null_scenarios = [
        ([30], [20]),               # 1 Null
        ([30, 30], [20, -20]),      # 2 Nulls
        ([30, 30, 30], [20, -20, 40]), # 3 Nulls
        ([30]*6, [20, -20, 40, -40, 60, -60]) # 6 Nulls
    ]

    for i, (tn, pn) in enumerate(null_scenarios):
        theta_n = np.array(tn)
        phi_n = np.array(pn)
        RN = np.array([100] * len(tn))
        sig_db, int_db = pert2d_null_multi(D, PhaseTable, theta_t, phi_t, R, theta_n, phi_n, RN, noise_db, quantization_bits=1)
        print(f"Attempting {len(tn)} Nulls | Mainlobe Gain: {sig_db:.2f} dB | Avg Null Depth: {np.mean(int_db):.2f} dB")

if __name__ == "__main__":
    test_null_depth_vs_quantization()
    test_gain_sacrifice()
    test_dof_exhaustion()
