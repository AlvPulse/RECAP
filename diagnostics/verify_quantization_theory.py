import numpy as np

def compute_af(theta_target, phi_target, theta_eval, phi_eval, bits, Nx=4, Ny=4, d=0.5):
    # k vectors
    k = 2 * np.pi

    # Target vector
    u_t = np.sin(np.radians(theta_target)) * np.cos(np.radians(phi_target))
    v_t = np.sin(np.radians(theta_target)) * np.sin(np.radians(phi_target))

    # Eval vector
    u_e = np.sin(np.radians(theta_eval)) * np.cos(np.radians(phi_eval))
    v_e = np.sin(np.radians(theta_eval)) * np.sin(np.radians(phi_eval))

    af_ideal = 0.0
    af_quant = 0.0

    for nx in range(Nx):
        for ny in range(Ny):
            x = nx * d
            y = ny * d

            # Ideal phase
            phase_ideal = -k * (x * u_t + y * v_t)

            # Quantize phase
            if bits == 1:
                # Phases: 0, pi
                phase_q = np.round(phase_ideal / np.pi) * np.pi
            elif bits == 2:
                # Phases: 0, pi/2, pi, 3pi/2
                phase_q = np.round(phase_ideal / (np.pi/2)) * (np.pi/2)
            else:
                phase_q = phase_ideal

            # Compute contributions
            w_ideal = np.exp(1j * phase_ideal)
            w_quant = np.exp(1j * phase_q)

            af_ideal += w_ideal * np.exp(1j * k * (x * u_e + y * v_e))
            af_quant += w_quant * np.exp(1j * k * (x * u_e + y * v_e))

    return np.abs(af_ideal), np.abs(af_quant)

# Test 1-bit mirror ambiguity
print("--- 1-Bit Ambiguity Test ---")
t_ideal, t_1bit = compute_af(30, 45, 30, 45, bits=1)
print(f"Target (30, 45) -> Ideal: {t_ideal:.2f}, 1-Bit: {t_1bit:.2f}")

m_ideal, m_1bit = compute_af(30, 45, 30, 45 + 180, bits=1)
print(f"Mirror (30, 225) -> Ideal: {m_ideal:.2f}, 1-Bit: {m_1bit:.2f}")

# Test 2-bit quantization lobes (approximate)
print("\n--- 2-Bit Quantization Lobe Test ---")
t_ideal, t_2bit = compute_af(30, 0, 30, 0, bits=2)
print(f"Target (30, 0) -> Ideal: {t_ideal:.2f}, 2-Bit: {t_2bit:.2f}")


print("\n--- Scanning for 2-bit Parasitic Lobes ---")
max_q_lobe = 0
max_q_angle = 0
for phi_e in range(0, 360, 5):
    if phi_e == 0: continue
    i_af, q_af = compute_af(30, 0, 30, phi_e, bits=2)
    if q_af > max_q_lobe and i_af < 5: # only look where ideal AF is low
        max_q_lobe = q_af
        max_q_angle = phi_e

print(f"Max 2-bit Parasitic Lobe found at phi={max_q_angle} with magnitude {max_q_lobe:.2f}")
