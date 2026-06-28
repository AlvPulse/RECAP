
import numpy as np
from .channel import total_path_loss, calculate_noise_level_db, landa

def array_locs(N):
    if N < 16 or N in [20, 24, 36, 60]:
        Ny = 1
    elif N < 64:
        Ny = 2
    elif N == 256:
        Ny = 8
    else:
        Ny = 4

    Nx = N // (4 * Ny)
    xAr = np.arange(-Nx, Nx + 1)
    yAr = np.arange(-Ny, Ny + 1)
    x = (xAr[:-1] + 0.5) / 2
    y = (yAr[:-1] + 0.5) / 2

    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()

    D = np.column_stack((X, Y))
    PhaseTable1 = np.tile([0, 180], (N, 1))

    added_phase_flag = np.mod(np.sum(D, axis=1) * 2, 2)
    added_phase = np.tile(added_phase_flag * 90, (2, 1)).T
    PhaseTable2 = PhaseTable1 + added_phase

    return D, PhaseTable2

def db(x):
    return 20 * np.log10(x)

def error_calculator(D, theta, phi, PhaseCode):
    K = 360
    Phase_wave_in = K * np.sin(np.radians(theta)) * D @ np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
    return np.mod(PhaseCode.astype(float) - Phase_wave_in, 360)

def _error_calculator_batch(D, thetas, phis, PhaseCode):
    """Vectorized error_calculator for multiple (theta, phi) directions simultaneously.
    thetas, phis: shape (M,); D: shape (N, 2); returns shape (M, N)."""
    K = 360
    st = np.sin(np.radians(thetas))           # (M,)
    cp = np.cos(np.radians(phis))             # (M,)
    sp = np.sin(np.radians(phis))             # (M,)
    # phase_wave_in[m, n] = K * sin(theta_m) * (D[n,0]*cos(phi_m) + D[n,1]*sin(phi_m))
    proj = D[:, 0][None, :] * cp[:, None] + D[:, 1][None, :] * sp[:, None]  # (M, N)
    phase_wave_in = K * st[:, None] * proj                                    # (M, N)
    return np.mod(PhaseCode[None, :].astype(float) - phase_wave_in, 360)

def _gain_batch(thetas, phis, PhaseCode, D):
    """Gain (dBi) at each of M directions simultaneously. Returns shape (M,)."""
    err = _error_calculator_batch(D, thetas, phis, PhaseCode)   # (M, N)
    AF  = np.abs(np.sum(np.exp(1j * np.deg2rad(err)), axis=1))  # (M,)
    return 20 * np.log10(np.maximum(AF, 1e-300))

def find_gain_of_tphi(theta, phi, PhaseCode, D):
    ErrorArray = error_calculator(D, theta, phi, PhaseCode)
    return db(np.abs(np.sum(np.exp(1j * np.deg2rad(ErrorArray)))))

def find_gain_of_tphi_n(theta, phi, PhaseCode, D):
    N = len(PhaseCode)
    ErrorArray = error_calculator(D, theta, phi, PhaseCode)
    Gain = db(np.abs(np.sum(np.exp(1j * np.deg2rad(ErrorArray)))))/2
    return Gain - db(N)/2

def phase_code_finder(D, PhaseTable, theta, phi):
    PhaseCode1 = PhaseTable[:, 0]
    ErrorArray = error_calculator(D, theta, phi, PhaseCode1)
    PhaseCode = PhaseCode1.copy()
    mask = (ErrorArray > 90) & (ErrorArray < 270)
    PhaseCode[mask] += 180
    return PhaseCode

def find_gain_of_tphi_i(thetaN, phiN, RN, PhaseCode, D):
    # Vectorized: all null directions at once
    GainN = _gain_batch(thetaN, phiN, PhaseCode, D) / 2   # (M,)
    return GainN - db(4 * RN * np.pi / landa)

def find_most_effective_null_multi(D, theta, phi, thetaN, phiN, WeightN, PhaseCode, MinNumber):
    ErrorArrayT = error_calculator(D, theta, phi, PhaseCode)
    PhaseMat = np.exp(1j * np.deg2rad(ErrorArrayT))
    Rval = np.real(np.sum(PhaseMat))
    I    = np.imag(np.sum(PhaseMat))

    if Rval == 0 and I == 0:
        return False, np.random.randint(len(PhaseCode))

    DeltaG = Rval * np.real(PhaseMat) + I * np.imag(PhaseMat)   # (N,)

    # Vectorized: compute gradients for all null directions at once
    errN = _error_calculator_batch(D, thetaN, phiN, PhaseCode)   # (M, N)
    PhaseMatN = np.exp(1j * np.deg2rad(errN))                     # (M, N)
    RvalN = np.real(np.sum(PhaseMatN, axis=1))                    # (M,)
    IN    = np.imag(np.sum(PhaseMatN, axis=1))                    # (M,)
    DeltaGN = RvalN[:, None] * np.real(PhaseMatN) + IN[:, None] * np.imag(PhaseMatN)  # (M, N)
    DeltaNSum = WeightN @ DeltaGN                                  # (N,)

    DeltaN = DeltaG - DeltaNSum
    sorted_indices = np.argsort(DeltaN)
    IndAlter = sorted_indices[MinNumber]
    flag = DeltaN[sorted_indices[MinNumber]] >= 0
    return flag, IndAlter

def fom_calc_null_multi_internal(D, PhaseCode, theta0, phi0, thetaN, phiN, RN, Noise_level):
    Gain0 = find_gain_of_tphi_n(theta0, phi0, PhaseCode, D)
    # Vectorized: gain at all null directions in one call
    GainNulls = _gain_batch(thetaN, phiN, PhaseCode, D) - db(RN)  # (M,)
    GainArr   = np.maximum(GainNulls, Noise_level)
    FoM = Gain0 + np.max(GainArr)
    return FoM, Gain0

def fom_calc_null_multi(D, PhaseCode, theta0, phi0, thetaN, phiN, RN, Noise_level):
    Gain0     = find_gain_of_tphi(theta0, phi0, PhaseCode, D)
    GainNulls = _gain_batch(thetaN, phiN, PhaseCode, D) - db(RN)  # (M,)
    GainArr   = np.maximum(GainNulls, Noise_level)
    FoM = Gain0 + np.max(GainArr)
    return FoM, Gain0

def pert2d_null_multi(D, PhaseTable, theta, phi, R, thetaN, phiN, RN, Noise_level):
    PhaseCodeStart = phase_code_finder(D, PhaseTable, theta, phi)

    FoMStart, _ = fom_calc_null_multi_internal(D, PhaseCodeStart, theta, phi, thetaN, phiN, RN, Noise_level)
    InterferenceStartdB = find_gain_of_tphi_i(thetaN, phiN, RN, PhaseCodeStart, D)
    WeightsN = 10 ** ((InterferenceStartdB - np.max(InterferenceStartdB)) / 5)

    PhaseCodeBest = PhaseCodeStart.copy()
    FoMBest = FoMStart
    IndNumber = 0   # fix: was 1, skipped the most effective null element on first iteration
    FoMTarget = Noise_level + 3
    N_elem = len(PhaseCodeBest)

    # Guard against catastrophic main-beam collapse (inherent 1-bit array limitation).
    # Allow at most 6 dB loss (|AF| ≥ half of initial) so null-forming never destroys the signal.
    AF0 = np.abs(np.sum(np.exp(1j * np.deg2rad(error_calculator(D, theta, phi, PhaseCodeStart)))))
    min_AF = AF0 / 2.0

    while FoMBest > FoMTarget:
        PhaseCodeAltering = PhaseCodeBest.copy()
        _, ElementIndex = find_most_effective_null_multi(
            D, theta, phi, thetaN, phiN, WeightsN, PhaseCodeAltering, IndNumber
        )
        PhaseCodeAltering[ElementIndex] += 180

        # Beam-loss guard: skip flips that collapse the main beam beyond 6 dB.
        AF = np.abs(np.sum(np.exp(1j * np.deg2rad(error_calculator(D, theta, phi, PhaseCodeAltering)))))
        if AF < min_AF:
            IndNumber += 1
            if IndNumber >= N_elem:
                break
            continue

        FOMAltering, _ = fom_calc_null_multi_internal(
            D, PhaseCodeAltering, theta, phi, thetaN, phiN, RN, Noise_level
        )

        if FOMAltering < FoMBest:
            PhaseCodeBest = PhaseCodeAltering.copy()
            FoMBest = FOMAltering
            IndNumber = 0   # reset to best element after each successful flip
        else:
            IndNumber += 1
            if IndNumber >= N_elem:
                break

    Signal = find_gain_of_tphi(theta, phi, PhaseCodeBest, D) - total_path_loss(R)
    Interference = _gain_batch(thetaN, phiN, PhaseCodeBest, D) - np.vectorize(total_path_loss)(RN)

    return Signal, Interference
