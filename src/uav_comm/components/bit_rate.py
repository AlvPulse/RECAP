import numpy as np

# 3GPP TS 38.214 Table 5.2.2.1-1 (4-bit CQI Table 1 for NR)
# Mapping of SINR to Spectral Efficiency (SE) in bits/s/Hz
# The SINR values are approximate thresholds commonly used for these CQI indices.
CQI_TABLE = [
    # (SINR_threshold_dB, Spectral_Efficiency)
    (-6.7, 0.1523),  # CQI 1
    (-4.7, 0.2344),  # CQI 2
    (-2.3, 0.3770),  # CQI 3
    (0.2, 0.6016),   # CQI 4
    (2.4, 0.8770),   # CQI 5
    (4.3, 1.1758),   # CQI 6
    (5.9, 1.4766),   # CQI 7
    (8.1, 1.9141),   # CQI 8
    (10.3, 2.4063),  # CQI 9
    (11.7, 2.7305),  # CQI 10
    (14.1, 3.3223),  # CQI 11
    (16.3, 3.9023),  # CQI 12
    (18.7, 4.5234),  # CQI 13
    (21.0, 5.1152),  # CQI 14
    (22.7, 5.5547)   # CQI 15
]

def sinr_to_spectral_efficiency(sinr_linear):
    """
    Convert SINR (linear) to Spectral Efficiency (bits/s/Hz) using 5G NR CQI mapping.
    """
    if sinr_linear <= 0:
        return 0.0
    sinr_db = 10 * np.log10(sinr_linear)

    # Find highest CQI index where SINR >= threshold
    se = 0.0
    for threshold_db, efficiency in CQI_TABLE:
        if sinr_db >= threshold_db:
            se = efficiency
        else:
            break
    return se

def calculate_bit_rate(sinr_linear, bandwidth_ghz):
    """
    Calculate bit rate (Gbps) given SINR and bandwidth.
    """
    se = sinr_to_spectral_efficiency(sinr_linear)
    # Bandwidth is in GHz, so bit rate is in Gbps
    bit_rate_gbps = se * bandwidth_ghz
    return bit_rate_gbps
