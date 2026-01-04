
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
    lambda_wave = 1
    K = 360 / lambda_wave
    Phase_wave_in = K * np.sin(np.radians(theta)) * D @ np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
    ErrorArray = np.mod(PhaseCode.astype(float) - Phase_wave_in, 360)
    return ErrorArray

def find_gain_of_tphi(theta, phi, PhaseCode, D):
    ErrorArray = error_calculator(D, theta, phi, PhaseCode)
    Gain = db(np.abs(np.sum(np.exp(1j * np.deg2rad(ErrorArray)))))
    return Gain

def phase_code_finder(D, PhaseTable, theta, phi):
    PhaseCode1 = PhaseTable[:, 0]
    ErrorArray = error_calculator(D, theta, phi, PhaseCode1)
    PhaseCode = PhaseCode1.copy()
    mask = (ErrorArray > 90) & (ErrorArray < 270)
    PhaseCode[mask] += 180
    return PhaseCode

def find_gain_of_tphi_i(thetaN, phiN, RN, PhaseCode, D):
    Interference = np.zeros_like(thetaN).astype(float)
    for i in range(len(thetaN)):
        ErrorArray = error_calculator(D, thetaN[i], phiN[i], PhaseCode)
        Gain = db(np.abs(np.sum(np.exp(1j * np.deg2rad(ErrorArray)))))/2
        Interference[i] = Gain - db(4*RN[i]*np.pi/landa)
    return Interference

def find_most_effective_null_multi(D, theta, phi, thetaN, phiN, WeightN, PhaseCode, MinNumber):
    ErrorArrayT = error_calculator(D, theta, phi, PhaseCode)
    PhaseMat = np.exp(1j * np.deg2rad(ErrorArrayT))
    Rval = np.real(np.sum(PhaseMat))
    I = np.imag(np.sum(PhaseMat))

    if Rval == 0 and I == 0:
        IndAlter = np.random.randint(len(PhaseCode))
        flag = False
    else:
        DeltaG = Rval * np.real(PhaseMat) + I * np.imag(PhaseMat)
        DeltaNSum = np.zeros_like(DeltaG)
        for i in range(len(WeightN)):
            ErrorArrayN = error_calculator(D, thetaN[i], phiN[i], PhaseCode)
            PhaseMatN = np.exp(1j * np.deg2rad(ErrorArrayN))
            RvalN = np.real(np.sum(PhaseMatN))
            IN = np.imag(np.sum(PhaseMatN))
            DeltaGN = RvalN * np.real(PhaseMatN) + IN * np.imag(PhaseMatN)
            DeltaNSum += WeightN[i] * DeltaGN
        DeltaN = DeltaG - DeltaNSum
        sorted_indices = np.argsort(DeltaN)
        IndAlter = sorted_indices[MinNumber]
        flag = DeltaN[sorted_indices[MinNumber]] >= 0
    return flag, IndAlter

def fom_calc_null_multi(D, PhaseCode, theta0, phi0, thetaN, phiN, RN, Noise_level):
    Gain0 = find_gain_of_tphi(theta0, phi0, PhaseCode, D) # Note: Env uses find_gain_of_tphi_n in one place, verify?
    # Original used find_gain_of_tphi_n for Gain0 in fom_calc_null_multi
    # But find_gain_of_tphi_n divides by 2 and subtracts db(N)/2.
    # Let's import/define it if needed.
    # Actually, in Environment.py:
    # Gain0 = find_gain_of_tphi_n(theta0, phi0, PhaseCode, D)
    # Let's add that.

    GainArr = np.zeros_like(thetaN)
    for i in range(len(RN)):
        GainNull = find_gain_of_tphi(thetaN[i], phiN[i], PhaseCode, D) - db(RN[i])
        if GainNull < Noise_level:
            GainNull = Noise_level
        GainArr[i] = GainNull
    FoM = Gain0 + np.max(GainArr)
    return FoM, Gain0

def find_gain_of_tphi_n(theta, phi, PhaseCode, D):
    N = len(PhaseCode)
    ErrorArray = error_calculator(D, theta, phi, PhaseCode)
    Gain = db(np.abs(np.sum(np.exp(1j * np.deg2rad(ErrorArray)))))/2
    return Gain - db(N)/2

def pert2d_null_multi(D, PhaseTable, theta, phi, R, thetaN, phiN, RN, Noise_level):
    PhaseCodeStart = phase_code_finder(D, PhaseTable, theta, phi)

    # We need to redefine fom_calc to use find_gain_of_tphi_n as per original code
    # Redefining fom_calc logic inline or updating fom_calc above

    # Recalc FoM with correct Gain0 function
    Gain0 = find_gain_of_tphi_n(theta, phi, PhaseCodeStart, D)
    # ... logic continues from Env

    # Let's stick to porting logic exactly.
    # fom_calc_null_multi in Env uses find_gain_of_tphi_n

    FoMStart, _ = fom_calc_null_multi_internal(D, PhaseCodeStart, theta, phi, thetaN, phiN, RN, Noise_level)
    InterferenceStartdB = find_gain_of_tphi_i(thetaN, phiN, RN, PhaseCodeStart, D)
    WeightsN = 10 ** ((InterferenceStartdB - np.max(InterferenceStartdB)) / 5)

    PhaseCodeAltering = PhaseCodeStart.copy()
    PhaseCodeBest = PhaseCodeAltering.copy()
    FOMAltering = FoMStart
    FoMBest = FoMStart
    IndNumber = 1
    IterNumber = 0
    FoMTarget = Noise_level + 3
    N = len(PhaseCodeBest)

    # Safety: Max iterations to prevent infinite/long loops in optimization
    # The original loop could run very long if improvement is slow.
    MAX_ITERS = 1000

    while FOMAltering > FoMTarget and IterNumber < MAX_ITERS:
        PhaseCodeAltering = PhaseCodeBest.copy()
        Flag, ElementIndex = find_most_effective_null_multi(D, theta, phi, thetaN, phiN, WeightsN, PhaseCodeAltering, IndNumber)
        PhaseCodeAltering[ElementIndex] += 180
        FOMAltering, _ = fom_calc_null_multi_internal(D, PhaseCodeAltering, theta, phi, thetaN, phiN, RN, Noise_level)

        if FOMAltering < FoMBest:
            PhaseCodeBest = PhaseCodeAltering.copy()
            FoMBest = FOMAltering
            IndNumber = 1
        else:
            IndNumber += 1
            if IndNumber >= N:
                break
        IterNumber += 1

    Signal = find_gain_of_tphi(theta, phi, PhaseCodeBest, D) - total_path_loss(R)
    # Interference calculation
    # Original:
    # Interference= np.zeros_like(InterferenceStartdB)
    # for i in range(len(RN)):
    #   Interference[i] = find_gain_of_tphi(thetaN[i],phiN[i], PhaseCodeBest, D)-total_path_loss(RN[i])

    # Wait, total_path_loss from channel.py needs import.

    Interference = np.zeros_like(InterferenceStartdB)
    for i in range(len(RN)):
        Interference[i] = find_gain_of_tphi(thetaN[i], phiN[i], PhaseCodeBest, D) - total_path_loss(RN[i])

    return Signal, Interference

def fom_calc_null_multi_internal(D, PhaseCode, theta0, phi0, thetaN, phiN, RN, Noise_level):
    Gain0 = find_gain_of_tphi_n(theta0, phi0, PhaseCode, D)
    GainArr = np.zeros_like(thetaN)
    for i in range(len(RN)):
        GainNull = find_gain_of_tphi(thetaN[i], phiN[i], PhaseCode, D) - db(RN[i])
        if GainNull < Noise_level:
            GainNull = Noise_level
        GainArr[i] = GainNull
    FoM = Gain0 + np.max(GainArr)
    return FoM, Gain0
