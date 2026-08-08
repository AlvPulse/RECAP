# Theoretical Framework: Physics-Informed Action Masking for Quantized UAV Arrays

## 1. The Core Concept and The Scalability Problem
In a multi-panel mmWave UAV setup, dynamic restriction of the action space (user selection) based on physical interference boundaries is critical. The problem of computing joint feasibility for $K$ arrays appears cursed by dimensionality. However, we escape this curse by establishing that **interference under quantized baseband control is deterministically bounded by predictable array physics.**

## 2. The Physics of Quantization Lobes and Angular Ambiguity
When dealing with low-cost arrays employing 1-bit or 2-bit phase shifters, standard digital beamforming (e.g., Zero-Forcing or MMSE) fails catastrophically in specific spatial regions. This failure is not random; it is governed by exact geometric principles.

### A. 1-Bit Angular Ambiguity (The Mirror Lobe)
With 1-bit quantization (phases constrained to $0$ and $\pi$), the excitation weights become purely real-valued (or anti-aligned). Mathematically, this forces the complex array factor $AF(\theta, \phi)$ to have Hermitian symmetry.
**The Physical Consequence:** For every main beam steered towards a target angle $(\theta, \phi)$, an exact "mirror" grating lobe of equal magnitude is unavoidably generated in the opposite direction space (e.g., $\phi \pm 180^\circ$).
**The Action Rule:** It is physically impossible to isolate two users if they lie on this mirror-ambiguity line. The PINN must enforce a hard bound: if $User_1$ and $User_2$ are separated by $\approx 180^\circ$ azimuth, the expected interference penalty approaches 1.0 (total failure).

### B. 2-Bit Quantization Sidelobes
With 2-bit quantization (phases constrained to $0, \pi/2, \pi, 3\pi/2$), the mirror lobe is broken, but the periodic phase rounding introduces specific, predictable parasitic quantization lobes.
**The Physical Consequence:** These lobes appear at predictable harmonic offsets from the main beam, depending on the array spacing $d$ and the steering angle. They set a strict, inescapable "Quantization Noise Floor" (typically -10 dB to -15 dB).
**The Action Rule:** Deep nulls (e.g., -30 dB) are physically impossible. The PINN will bound the maximum pairwise isolation based on this noise floor. If User B lies anywhere near one of the primary quantization harmonic angles of User A, the PINN must penalize the selection.

## 3. The Rank Exhaustion Constraint (DoF)
Beyond spatial ambiguity, small arrays suffer from **Degrees of Freedom (DoF) Exhaustion**. An $N$-element array mathematically has $N-1$ degrees of freedom. Forcing it to null multiple users under coarse quantization rapidly exhausts this budget, collapsing the mainlobe regardless of angular separation.

**The Action Rule:** A cardinality limit must be enforced. The model will cap the maximum number of simultaneously served users in a given sector before the joint SINR fundamentally collapses.

## 4. Constructing the PINN Surrogate Model
Instead of full joint simulations, we train a Physics-Informed Neural Network (PINN) to act as a pairwise surrogate model.
*   **Inputs:** $(\Delta \theta, \Delta \phi, R_1/R_2, \text{Quantization\_Bits})$
*   **Physics Loss Functions:** The PINN loss includes mathematical regularizers:
    *   `Loss_Mirror`: Enforces $Penalty(\Delta \phi \approx 180^\circ | \text{1-bit}) = 1.0$
    *   `Loss_HPBW`: Enforces $Penalty(\Delta \phi < \text{HPBW}) = 1.0$
    *   `Loss_NoiseFloor`: Enforces $Penalty \ge \text{Quantization\_Noise\_Floor}$
*   **Output:** Joint SINR Retention Score $\in [0, 1]$.

By using this PINN, the RL agent operates entirely aware of the hardware's exact physical limits without running computationally expensive baseband simulations during training.
