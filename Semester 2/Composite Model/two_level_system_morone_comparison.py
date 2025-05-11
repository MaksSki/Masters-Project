import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# Parameters
collision_num = 1000
record_step = 1
g = 1
omega_0 = 1
theta = np.pi / 4
q = 0.75
kappa = 1
delta_t = 1
c = 1
p_values = [0.8, 0.85, 0.9, 0.95, 0.75, 0.7,1]

# Define fresh ancilla (charger)
def fresh_ancilla():
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q += coherence_term
    return rho_Q.unit()

# Partial swap quantum map
def partial_swap_map(rho, p):
    swap_op = swap(2, 2)
    return ((1 - p) * rho + p * swap_op * rho * swap_op.dag()).unit()

# Morrone et al. ergotropy definition (for qubit)
def morrone_ergotropy(rho, omega_0=1.0):
    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    exp_x, exp_y, exp_z = expect(sx, rho), expect(sy, rho), expect(sz, rho)
    sigma_norm = np.sqrt(exp_x**2 + exp_y**2 + exp_z**2)
    return (omega_0 / 2) * (sigma_norm + exp_z)

# Initial battery state
battery = ket2dm(basis(2, 0))

# Interaction Unitaries
V = g * (tensor(sigmam(), sigmap()) + tensor(sigmap(), sigmam()))
U = (-1j * theta * V).expm()

V_env = np.sqrt(kappa / delta_t) * (tensor(sigmam(), sigmap()) + tensor(sigmap(), sigmam()))
U_env = (-1j * V_env).expm()

# Simulation
plt.figure(figsize=(10, 6))

for p_swap in p_values:
    prev_env_ancilla = fresh_ancilla()
    current_state = tensor(battery, fresh_ancilla())

    ergotropy_vals = []
    collisions = []

    for collision in range(1, collision_num + 1):
        # Battery-Charger interaction
        current_state = U * current_state * U.dag()
        battery_state = current_state.ptrace(0)

        # Partial swap between ancillas
        combined_ancillas = tensor(fresh_ancilla(), prev_env_ancilla)
        swapped_ancillas = partial_swap_map(combined_ancillas, p_swap)
        new_env_ancilla = swapped_ancillas.ptrace(0)

        # Battery-Environment interaction
        env_interaction = tensor(battery_state, new_env_ancilla)
        env_interaction = U_env * env_interaction * U_env.dag()

        battery_state = env_interaction.ptrace(0)
        prev_env_ancilla = env_interaction.ptrace(1)

        # Record ergotropy using Morrone definition
        ergotropy_vals.append(morrone_ergotropy(battery_state, omega_0))
        collisions.append(collision)

        # Reset for next collision
        current_state = tensor(battery_state, fresh_ancilla())

    plt.plot(collisions, ergotropy_vals, label=f"p={p_swap}", linewidth=2)

# Plot adjustments
plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Ergotropy (Morrone definition)", fontsize=14)
plt.title("Ergotropy vs. Number of Collisions (Two-Level System)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
