import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from typing import Sequence

# ────────────────────────────────────────────────────────────────────────────────
# Model parameters (keep these identical to the original unless you want speed-ups)
# ────────────────────────────────────────────────────────────────────────────────
N              = 200           # battery dimension (ℏω = 1 units)
collision_num  = 1000          # number of battery–ancilla collisions
record_step    = 1             # not used in this script, but kept for parity
g              = 1.0           # Jaynes–Cummings coupling strength
theta          = np.pi / 4     # battery–ancilla interaction angle
kappa          = 1.0           # system–environment interaction strength
delta_t        = 1.0           # collision time (sets γ = κ/Δt)
q              = 0.75          # fresh ancilla excited-state population
c              = 1.0           # coherence prefactor in ρᵠ
p_values       = (0.05, 1)    # **only** the two requested swap probabilities

# Temperature sweep (K)
T_min, T_max, T_points = 0.1, 100.0, 100
T_grid: Sequence[float] = np.linspace(T_min, T_max, T_points)

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────

def fresh_ancilla(q: float = q, c: float = c) -> Qobj:
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    return (rho_Q + coherence_term)

def realistic_env_state(temperature: float, omega: float = 1.0, coherence_strength: float = 0.1) -> Qobj:
    k_B = 0.01
    beta = 1.0 / (k_B * temperature)
    Z = np.exp(-beta * omega / 2) + np.exp(beta * omega / 2)
    p0, p1 = np.exp(-beta * omega / 2) / Z, np.exp(beta * omega / 2) / Z
    max_coh = np.sqrt(p0 * p1)
    coh = min(coherence_strength, max_coh)
    return Qobj([[p0, coh], [coh, p1]])

def get_passive_energy(rho: Qobj, levels: np.ndarray) -> float:
    eigvals, _ = rho.eigenstates()
    sorted_pop = np.sort(np.array(eigvals, dtype=float))[::-1]
    return float(np.dot(sorted_pop, levels))

def peak_ergotropy(T: float, p_swap: float, N_bat: int = N, collisions: int = collision_num) -> float:
    A = sum(basis(N_bat + 1, n - 1) * basis(N_bat + 1, n).dag() for n in range(1, N_bat + 1))
    A_dag = A.dag()

    V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U = (-1j * theta * V).expm()
    V_env = np.sqrt(kappa / delta_t) * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U_env = (-1j * V_env).expm()

    battery_int = basis(N_bat + 1, 0)
    one_state    = ket2dm(tensor(battery_int, basis(2, 1)))
    prev_env_anc = realistic_env_state(T)

    levels = np.arange(N_bat + 1)
    peak_E = 0.0

    for _ in range(collisions):
        one_state = U * one_state * U.dag()
        battery   = one_state.ptrace(0)

        combined    = tensor(realistic_env_state(T), prev_env_anc)
        swap_op     = swap(2, 2)
        swapped     = ((1 - p_swap) * combined + p_swap * swap_op * combined * swap_op.dag()).unit()
        new_env_anc = swapped.ptrace(0)

        sys_env      = tensor(battery, new_env_anc)
        sys_env      = U_env * sys_env * U_env.dag()
        battery      = sys_env.ptrace(0)
        prev_env_anc = sys_env.ptrace(1)

        avg_E  = float(np.dot(levels, battery.diag()))
        ergo_E = avg_E - get_passive_energy(battery, levels)
        peak_E = max(peak_E, ergo_E)

        one_state = tensor(battery, fresh_ancilla())

    return peak_E

# ────────────────────────────────────────────────────────────────────────────────
# Main sweep
# ────────────────────────────────────────────────────────────────────────────────
non_markov = []
for T in T_grid:
    peak_0   = peak_ergotropy(T, p_swap=0.0)
    peak_08  = peak_ergotropy(T, p_swap=0.8)
    non_markov.append(peak_08 / peak_0 if peak_0 else np.nan)

# ────────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 9))

# Set global font size to 18
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

plt.plot(T_grid, non_markov)
plt.xlabel("Temperature (Kelvin) ", fontsize=18)
plt.ylabel("Non-Markovianity ", fontsize=18)
plt.title("Non-Markovianity vs Temperature", fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()
