# Theoretical Framework: Physics-Informed Dynamic Action Masking for mmWave UAVs

## 1. The Core Concept and The Scalability Problem
In a multi-panel mmWave UAV setup, we want to dynamically restrict the action space (which user each panel targets) based on physical interference boundaries. The problem seems cursed by dimensionality: Panel $i$'s ability to serve User $u$ while nulling interference toward all $K-1$ users chosen by other panels depends on the joint configuration. Tabulating or learning this joint feasibility appears exponential in $K$.

## 2. The Theoretical Anchor: Mask-Induced Decoupling Lemma
We escape the curse of dimensionality using classical physics and linear algebra.

For null constraints stacked in matrix $V$, the achievable gain toward $u$ is the projection $\|P_V^\perp a_u\|^2$. The coupling between constraints enters only through the **Gram matrix of the steering vectors**.

1. **Mainlobe–Null Proximity is First-Order:** Gain collapses when a null direction is angularly close to the serve direction (pairwise proximity).
2. **Null–Null Proximity is Benign (Second-Order):** If two null directions are close, their steering vectors are collinear—one spatial null covers both. The joint problem gets easier. Severely coupled regimes only occur at small separations, which our mask strictly forbids.

**The Lemma:** *If the pairwise mask enforces angular separation beyond the Half-Power Beamwidth (HPBW) between every serve direction and every null direction, the steering-vector Gram matrix is diagonally dominant. Joint feasibility deviates from the product of pairwise feasibilities by a bounded $\epsilon$ (a Gershgorin-type bound).*

**Conclusion:** The mask creates the very conditions under which a simple pairwise approximation is valid. We do not suffer the curse of dimensionality *because* we mask.

## 3. The Architecture: M-D Generator & E-C Executor

### The Mask Generator (M-D: Structured Hybrid)
The mask is a union of a deterministic core and learned residual regions:
1. **M-A (Deterministic Core):** A pairwise feasibility table based on the geometric array factor (HPBW + first-null width).
2. **M-B (Violation-Driven Growth):** The mask grows to absorb residual second-order joint effects that the pairwise table misses (e.g., quantization-lobe interactions due to 1-bit or 2-bit phase shifters).
3. **Anti-Ratchet ($\epsilon$-Probing):** To prevent "Capacity Collapse" (where the mask ratchets outward indefinitely and permanently shrinks the feasible space), we use asymmetric $\epsilon$-probing. We explicitly relax one mask cell at a time in low-stakes slots to test if the boundary can be shrunk safely.

### The Executor (E-C: One-Shot Conservative Masking with Repair)
Standard PPO per-head masks cannot express joint conditional constraints (e.g., "User $u$ is forbidden for Panel $i$ *only if* Panel $j$ picks User $v$").
1. **One-Shot Masking:** We mask the *union* of regions that any head could trigger.
2. **Deterministic Repair:** Because sampled joint actions can still occasionally collide due to expressiveness gaps, we implement a deterministic repair step. If a joint action contains a conflict, a fixed-priority order reassigns the lower-priority panel to its best unmasked alternative.
3. **Validation:** If the Decoupling Lemma holds, the repair is almost never invoked. We log the empirical **conflict rate** as direct proof of the lemma.
