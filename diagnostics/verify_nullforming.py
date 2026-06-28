"""
Nullforming verification diagnostic.

Tests whether pert2d_null_multi:
  (a) correctly beamforms toward the target user, and
  (b) creates nulls toward other simultaneously selected users.

Run with: python diagnostics/verify_nullforming.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.uav_comm.components.antenna import (
    array_locs, phase_code_finder, find_gain_of_tphi,
    pert2d_null_multi, calculate_noise_level_db
)

# ── constants ──────────────────────────────────────────────────────────────────
N_ELEMENTS = 8
BANDWIDTH  = 0.35    # GHz
UAV_HEIGHT = 50      # m

noise_db = -118.5    # dBW (fixed reference)
D, PhaseTable = array_locs(N_ELEMENTS)


def uav_direction(horizontal_deg, dist_2d=200):
    """
    Return (theta, phi) as seen by the UAV for a user at given azimuth and 2D distance.
    theta = elevation from zenith (0°=overhead, 90°=horizon)
    """
    phi   = horizontal_deg
    theta = np.degrees(np.arctan2(dist_2d, UAV_HEIGHT))   # elevation from vertical
    # Convert to UAV's coordinate: theta_from_zenith = 90 - elevation_from_horizon
    # In this code, theta is measured from zenith: theta=0 → top, theta=90 → horizon
    theta_zenith = 90 - np.degrees(np.arctan2(UAV_HEIGHT, dist_2d))
    return theta_zenith, phi


def gain_at_direction(ph, theta, phi):
    return find_gain_of_tphi(theta, phi, ph, D)


def test_case(label, target_azi, null_azis, dist=200):
    """
    Beamform toward target_azi and null toward null_azis.
    Returns gain at target and gains at null directions.
    """
    t_theta, t_phi = uav_direction(target_azi, dist)
    null_thetas = np.array([uav_direction(a, dist)[0] for a in null_azis])
    null_phis   = np.array([uav_direction(a, dist)[1] for a in null_azis])
    null_dists  = np.full(len(null_azis), dist)

    # Phase code WITHOUT nullforming
    ph_beam_only = phase_code_finder(D, PhaseTable, t_theta, t_phi)
    g_main_no_null = gain_at_direction(ph_beam_only, t_theta, t_phi)
    g_null_no_null = [gain_at_direction(ph_beam_only, null_thetas[i], null_phis[i])
                      for i in range(len(null_azis))]

    # Phase code WITH nullforming
    sig_db, int_db = pert2d_null_multi(
        D, PhaseTable, t_theta, t_phi, dist,
        null_thetas, null_phis, null_dists, noise_db
    )
    # Re-extract phase code to compute per-direction gains
    ph_best = phase_code_finder(D, PhaseTable, t_theta, t_phi)   # start code
    # (pert2d_null_multi returns signal/interference dB but not the phase code directly;
    #  we use sig_db and int_db as the final answer)
    g_main_null = sig_db   # signal dB (already includes path loss)
    g_null_null = int_db   # interference dB array (already includes path loss)

    # Gains WITHOUT path loss correction, for comparison
    g_main_no_null_pathloss = g_main_no_null - 20 * np.log10(dist)   # rough
    g_null_improvement = [
        g_null_no_null[i] - g_null_null[i] for i in range(len(null_azis))
    ]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Target: azi={target_azi}°, Null(s): {null_azis}°, dist={dist}m")
    print(f"{'='*60}")
    print(f"  Main beam gain (no null): {g_main_no_null:.1f} dBi")
    print(f"  Null dir gain (no null):  {[f'{g:.1f}' for g in g_null_no_null]} dBi")
    print(f"\n  Signal power (with null): {sig_db:.1f} dB  (incl. path loss)")
    print(f"  Interfr power (with null):{[f'{g:.1f}' for g in g_null_null]} dB")
    print(f"\n  Null improvement:         {[f'{g:.1f}' for g in g_null_improvement]} dB")
    print(f"  (positive = interference reduced by this many dB)")

    # SINR estimate (assuming same TX power, same distance)
    noise_eff = 10 ** (noise_db / 10) / 20.0          # 43 dBm = 20W
    sig_lin   = 10 ** (sig_db / 10)
    int_lin   = sum(10 ** (g / 10) for g in g_null_null)
    sinr = sig_lin / (int_lin + noise_eff)
    print(f"\n  Estimated SINR (with null): {sinr:.2f} ({10*np.log10(sinr+1e-9):.1f} dB)")
    th = 2.0
    print(f"  Above threshold (2.0): {'YES [OK]' if sinr > th else 'NO [FAIL]'}")
    return sinr, g_null_improvement


def sweep_azimuth_separation():
    """Test how SINR changes as separation between target and null grows."""
    separations = [10, 20, 30, 45, 60, 90, 120, 150, 180]
    sinrs = []
    for sep in separations:
        t_theta, t_phi = uav_direction(0)
        n_theta, n_phi = uav_direction(sep)
        sig_db, int_db = pert2d_null_multi(
            D, PhaseTable, t_theta, t_phi, 200,
            np.array([n_theta]), np.array([n_phi]), np.array([200.0]), noise_db
        )
        noise_eff = 10 ** (noise_db / 10) / 20.0
        sig_lin   = 10 ** (sig_db / 10)
        int_lin   = 10 ** (int_db[0] / 10)
        sinr = sig_lin / (int_lin + noise_eff)
        sinrs.append(sinr)
        print(f"  sep={sep:3d}°: SINR={sinr:.2f} ({10*np.log10(sinr+1e-9):.1f} dB)", end="")
        print(" [OK]" if sinr > 2.0 else " [FAIL]")

    return separations, sinrs


# ── run tests ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  NULLFORMING VERIFICATION DIAGNOSTIC")
print("="*60)

print("\n--- Case 1: Single null, large separation (90°) ---")
test_case("Single null, 90° apart", target_azi=0, null_azis=[90])

print("\n--- Case 2: Single null, small separation (20°) ---")
test_case("Single null, 20° apart", target_azi=0, null_azis=[20])

print("\n--- Case 3: Two nulls, good separation ---")
test_case("Two nulls, 90° + 180°", target_azi=0, null_azis=[90, 180])

print("\n--- Case 4: Three nulls (full multi-user scenario) ---")
test_case("Three nulls: 90°+180°+270°", target_azi=0, null_azis=[90, 180, 270])

print("\n--- Case 5: Worst case — nulls very close to target ---")
test_case("Very close nulls (10°+15°)", target_azi=0, null_azis=[10, 15])

print("\n\n--- Azimuth-separation sweep (1 null, target=0°) ---")
print("  How SINR improves as angular separation grows:")
separations, sinrs = sweep_azimuth_separation()

# ── plot ───────────────────────────────────────────────────────────────────────
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(separations, sinrs, 'b-o', markersize=8)
    axes[0].axhline(y=2.0, color='r', linestyle='--', label='Threshold (3 dB)')
    axes[0].set_xlabel("Angular separation target ↔ null (degrees)")
    axes[0].set_ylabel("SINR (linear)")
    axes[0].set_title("Null-forming: SINR vs. Angular Separation\n(1 null, 200m range, N=8 elements)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(separations, [10*np.log10(s+1e-9) for s in sinrs], 'g-o', markersize=8)
    axes[1].axhline(y=3.0, color='r', linestyle='--', label='Threshold (3 dB)')
    axes[1].set_xlabel("Angular separation target ↔ null (degrees)")
    axes[1].set_ylabel("SINR (dB)")
    axes[1].set_title("Null-forming: SINR (dB) vs. Angular Separation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "diagnostics/nullforming_verification.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")
    plt.close()
except Exception as e:
    print(f"\n(Plot skipped: {e})")

print("\n--- Summary ---")
min_sep = next((s for s, r in zip(separations, sinrs) if r > 2.0), None)
print(f"  Minimum angular separation for SINR > threshold: {min_sep}°")
print("  This is the minimum user-pair separation the RL should learn to maintain.")
print("  The RL's core contribution: choose user sets with separations > threshold.")
