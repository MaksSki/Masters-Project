# Varying Interaction strength with the enviroment over different 

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import time

# Parameters
collision_num = 1000
record_step = 1
g = 1
N = 200
theta = np.pi / 4
q = 0.75
# default kappa is no longer used; we loop over multiple kappas
delta_t = 1
c = 1
p_values = [0, 0.2, 0.4, 0.6, 0.8, 1]  # Different partial swap strengths
temperature = 0.05
start = time.time()

# Set up a list of kappas to simulate
kappa_values = np.linspace(0.1,4,10)

def fresh_ancilla():
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q += coherence_term
    return rho_Q.unit()

# Realistic partially coherent thermal state for environment ancilla
def realistic_env_state(omega=1.0, coherence_strength=0.1, temperature=temperature):
    k_B = 1.0
    beta = 1 / (k_B * temperature)
    Z = np.exp(-beta * omega / 2) + np.exp(beta * omega / 2)

    p0 = np.exp(-beta * omega / 2) / Z  # population of |0⟩
    p1 = np.exp(beta * omega / 2) / Z   # population of |1⟩

    # Maximum allowed coherence to keep rho physical:
    max_coh = np.sqrt(p0 * p1)
    coh = min(coherence_strength, max_coh)

    rho = Qobj([[p0, coh], [coh, p1]])
    return rho

def partial_swap_map(rho, p):
    swap_op = swap(2, 2)
    rho_new = (1 - p) * rho + p * swap_op * rho * swap_op.dag()
    return rho_new.unit()

def get_passive_energy(rho, levels):
    eigvals, _ = rho.eigenstates()
    eigvals = np.array(eigvals, dtype=float)
    sorted_populations = np.sort(eigvals)[::-1]
    passive_energy = np.dot(sorted_populations, levels)
    return passive_energy

# Initial battery state
battery_int = basis(N+1, 0)

# Battery ladder operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()

# Interaction Unitary for battery dynamics (kept fixed)
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()

# Prepare plot for ergotropy vs. p
plt.figure(figsize=(10, 6))

levels = np.arange(N+1)

# Loop over kappa values; each will yield a curve (ergotropy vs. p)
for kappa in kappa_values:
    # Prepare an array to hold final ergotropy for each p value after all collisions
    final_ergotropy = []
    for p_swap in p_values:
        prev_env_ancilla = realistic_env_state()
        one_state = ket2dm(tensor(battery_int, basis(2, 1)))
        # We will record the ergotropy at the final collision for each p_swap.
        for collision in range(1, collision_num + 1):
            one_state = U * one_state * U.dag()
            battery = one_state.ptrace(0)

            # Environment collision using the partial swap map
            combined = tensor(realistic_env_state(), prev_env_ancilla)
            swapped = partial_swap_map(combined, p_swap)
            new_env_ancilla = swapped.ptrace(0)

            one_state_env = tensor(battery, new_env_ancilla)
            # Now the interaction with the environment uses kappa, so we redefine U_env accordingly:
            V_env = np.sqrt(kappa / delta_t) * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
            U_env = (-1j * V_env).expm()

            one_state_env = U_env * one_state_env * U_env.dag()

            battery = one_state_env.ptrace(0)
            updated_env_ancilla = one_state_env.ptrace(1)
            prev_env_ancilla = updated_env_ancilla

            # Update one_state for next collision
            one_state = tensor(battery, fresh_ancilla())

        # Compute final ergotropy for this p_swap
        probability = battery.diag()
        average_energy = np.dot(levels, probability)
        ergotropy = average_energy - get_passive_energy(battery, levels)
        final_ergotropy.append(ergotropy)
    
    # Plot final ergotropy vs. p for this kappa value
    plt.plot(p_values, final_ergotropy, marker='o', linewidth=2, label=f"kappa={kappa}")

plt.xlabel("p (partial swap strength)", fontsize=14)
plt.ylabel("Ergotropy", fontsize=14)
plt.title(f"Ergotropy vs. p for different kappa values (T={temperature})", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

end = time.time()
print("Time Ellapsed:", end - start)
print("In minutes:", (end - start) / 60)
