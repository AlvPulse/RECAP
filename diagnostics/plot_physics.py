import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from src.uav_comm.components.antenna import array_locs, pert2d_null_multi, phase_code_finder, find_gain_of_tphi

D, PhaseTable = array_locs(8)
theta_t = 30
phi_t = 0
R = 100
RN = np.array([100])
noise_db = -90
bandwidth = 0.35

ph_baseline = phase_code_finder(D, PhaseTable, theta_t, phi_t, quantization_bits=1)
gain_baseline_db = find_gain_of_tphi(theta_t, phi_t, ph_baseline, D)
sig_lin_base = 10 ** (gain_baseline_db / 10)
noise_lin = 10 ** (noise_db / 10)
sinr_base = sig_lin_base / noise_lin
cap_base = bandwidth * np.log2(1 + sinr_base)

angles = np.linspace(0, 360, 360)
true_retention = []

for phi_int in angles:
    sig_db, int_db = pert2d_null_multi(D, PhaseTable, theta_t, phi_t, R, np.array([30]), np.array([phi_int]), RN, noise_db, quantization_bits=1)
    sig_lin = 10 ** (sig_db / 10)
    int_lin = 10 ** (int_db[0] / 10)
    sinr = sig_lin / (int_lin + noise_lin)
    cap = bandwidth * np.log2(1 + sinr)
    true_retention.append(cap / cap_base)

plt.plot(angles, true_retention)
plt.title('True Capacity Retention (1-Bit)')
plt.savefig('diagnostics/true_retention_1bit.png')
