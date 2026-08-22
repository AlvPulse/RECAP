import numpy as np
import torch
import torch.nn as nn

class PairwiseInterferencePINN(nn.Module):
    """
    Physics-Informed Neural Network Surrogate Model for Quantized Array Interference.
    Predicts the expected SINR Retention Score [0, 1] when serving two users simultaneously.
    """
    def __init__(self, hpbw_deg=20.0, quantization_bits=1):
        super(PairwiseInterferencePINN, self).__init__()
        self.hpbw_deg = hpbw_deg
        self.quantization_bits = quantization_bits

        # In a real scenario, this would be a trained MLP.
        # For now, we embed the explicit physics rules derived from our analytical tests.
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

        # The theoretical noise floor limit based on quantization bits
        # 1-bit -> ~ -10 dB -> 0.1 linear retention limit for deep nulls
        # 2-bit -> ~ -15 dB -> 0.03 linear retention limit
        self.noise_floor_penalty = 0.1 if quantization_bits == 1 else 0.03

    def forward(self, delta_phi, distance_ratio):
        # Normalize inputs
        dp = torch.abs(delta_phi) % 360.0
        dp = torch.minimum(dp, 360.0 - dp) # Map to [0, 180]

        # Rule 1: HPBW Coupling Bound
        # The true physics engine collapses capacity very aggressively.
        # Using a sharper slope (4.0 instead of 2.0) and shifting the midpoint
        # to effectively capture the wide sidelobe interference penalty.
        hpbw_mask = torch.sigmoid((dp - (self.hpbw_deg * 1.5)) * 4.0)

        # Rule 2: 1-Bit Mirror Ambiguity
        if self.quantization_bits == 1:
            mirror_diff = torch.abs(dp - 180.0)
            mirror_mask = torch.sigmoid((mirror_diff - (self.hpbw_deg * 1.5)) * 4.0)

            # The base capacity retention for 1-bit arrays across random angles is extremely low
            # (often < 0.1) due to the massive -10dB sidelobe plateau.
            # We scale the maximum possible retention down severely.
            base_retention_max = 0.15
            score = hpbw_mask * mirror_mask * base_retention_max
        else:
            # Rule 3: 2-Bit Harmonic Lobe (Approx at 50 deg from tests)
            harmonic_diff = torch.abs(dp - 50.0)
            harmonic_mask = torch.sigmoid((harmonic_diff - self.hpbw_deg) * 4.0)

            base_retention_max = 0.40 # 2-bit is better, but still leaks heavily
            # Harmonic lobe creates a severe dip in the retention
            score = hpbw_mask * (harmonic_mask * 0.8 + 0.2) * base_retention_max

        # Rule 4: Distance coupling & Quantization Noise Floor
        # Even at perfect angles, noise floor limits isolation
        isolation_limit = 1.0 - (self.noise_floor_penalty * distance_ratio)
        isolation_limit = torch.clamp(isolation_limit, 0.0, 1.0)

        final_score = score * isolation_limit
        return final_score

    def get_interference_matrix(self, azimuths, distances):
        """
        Generates the NxN Interference Penalty Matrix using the PINN logic.
        azimuths: array of user angles in degrees
        distances: array of user distances from UAV
        Returns: NxN matrix where 1.0 = Perfect Isolation, 0.0 = Total Interference
        """
        N = len(azimuths)
        matrix = np.ones((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(i+1, N):
                d_phi = torch.tensor([azimuths[i] - azimuths[j]], dtype=torch.float32)
                dist_ratio = torch.tensor([max(distances[i], distances[j]) / (min(distances[i], distances[j]) + 1e-6)], dtype=torch.float32)

                score = self.forward(d_phi, dist_ratio).item()
                matrix[i, j] = score
                matrix[j, i] = score

        return matrix
