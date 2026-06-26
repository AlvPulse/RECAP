
import numpy as np

# Constants
BANDWIDTH = 0.35  # 350 MHz
f = 28  # GHz
c = 0.3
landa = c / f

def calculate_noise_level_db(bandwidth_ghz, temperature_kelvin=290):
    k = 1.38e-23
    bandwidth_hz = bandwidth_ghz * 1e9
    noise_power_watts = k * temperature_kelvin * bandwidth_hz
    noise_level_db = 10 * np.log10(noise_power_watts)
    return noise_level_db

def rayleigh_fading():
    return np.random.rayleigh(scale=1.0)

def path_loss_los(d_n_m, f_c, zeta_0, zeta_1, zeta_2):
    return zeta_0 + zeta_1 * np.log10(d_n_m) + zeta_2 * np.log10(f_c)

def path_loss_nlos(d_n_m, f_c, eta_0, eta_1, eta_2):
    return eta_0 + eta_1 * np.log10(d_n_m) + eta_2 * np.log10(f_c)

def los_probability_uav(d, h_uav, h_user=1.5, a=12.08, b=0.11):
    h_diff = h_uav - h_user
    theta_deg = np.degrees(np.arctan2(h_diff, d))
    P_LoS = 1.0 / (1.0 + a * np.exp(-b * (theta_deg - a)))
    return np.clip(P_LoS, 0.0, 1.0)

def total_path_loss(d_n_m, h_n=50, h_m=1.5, f_c=28, fading=False):
    # fading=False by default for training stability.
    # When re-enabling stochastic fading, use: 10*log10(h²) where h ~ Rayleigh(σ=1/√2)
    # so that mean channel power E[h²] = 1 (0 dB shift on average).
    zeta_params = (38.77, 16.7, 18.2)
    eta_params = (36.85, 30, 18.9)

    P_LoS = los_probability_uav(d_n_m, h_n)
    PL_LoS = path_loss_los(d_n_m, f_c, *zeta_params)
    PL_NLoS = path_loss_nlos(d_n_m, f_c, *eta_params)

    total_PL = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS

    if fading:
        h = rayleigh_fading()                          # amplitude, Rayleigh(scale=1/√2)
        fading_db = 10 * np.log10(max(h ** 2, 1e-12)) # power in dB; subtract from path loss (gain reduces loss)
        return total_PL - fading_db

    return total_PL
