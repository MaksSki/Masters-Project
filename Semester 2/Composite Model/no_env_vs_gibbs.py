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
kappa = 1
delta_t = 1
c = 1
p_values = [0, .2, .4, 0.6, 0.8, 1]  # Different partial swap strengths

start = time.time()

def fresh_ancilla():
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q += coherence_term
    return rho_Q

# Realistic partially coherent thermal state for environment ancilla
def realistic_env_state(omega=1.0, temperature=1, coherence_strength=0.1):
    k_B = 1.0
    beta = 1 / (k_B * temperature)
    Z = np.exp(-beta * omega / 2) + np.exp(beta * omega / 2)
    rho_thermal = Qobj([
        [np.exp(-beta * omega / 2) / Z, 0],
        [0, np.exp(beta * omega / 2) / Z]
    ])
    coherence_term = coherence_strength * (basis(2, 0) * basis(2, 1).dag() +
                                           basis(2, 1) * basis(2, 0).dag())
    rho_env = rho_thermal + coherence_term
    return rho_env

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

# Interaction Unitaries
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()

V_env = np.sqrt(kappa / delta_t) * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U_env = (-1j * V_env).expm()

# Prepare plot
plt.figure(figsize=(10, 9))

# Plot with environment
for p_swap in p_values:
    prev_env_ancilla = realistic_env_state()
    one_state = ket2dm(tensor(battery_int, basis(2, 1)))
    avg_energy_battery = []
    collisions = []
    ergo_val = []
    levels = np.arange(N+1)

    for collision in range(1, collision_num + 1):
        one_state = U * one_state * U.dag()
        battery = one_state.ptrace(0)

        combined = tensor(realistic_env_state(), prev_env_ancilla)
        swapped = partial_swap_map(combined, p_swap)
        new_env_ancilla = swapped.ptrace(0)

        one_state_env = tensor(battery, new_env_ancilla)
        one_state_env = U_env * one_state_env * U_env.dag()

        battery = one_state_env.ptrace(0)
        updated_env_ancilla = one_state_env.ptrace(1)
        prev_env_ancilla = updated_env_ancilla
        
        

        probability = battery.diag()
        average = np.dot(levels, probability)
        ergotropy = average - get_passive_energy(battery, levels)
        avg_energy_battery.append(average)
        ergo_val.append(ergotropy)
        collisions.append(collision)

        one_state = tensor(battery, fresh_ancilla())

    plt.plot(collisions, ergo_val, label=f"p={p_swap}", linewidth=2)

# Plot without environment
one_state = ket2dm(tensor(battery_int, basis(2, 1)))
ergo_val_no_env = []
collisions_no_env = []

for collision in range(1, collision_num + 1):
    one_state = U * one_state * U.dag()
    battery = one_state.ptrace(0)

    probability = battery.diag()
    average = np.dot(levels, probability)
    ergotropy = average - get_passive_energy(battery, levels)
    ergo_val_no_env.append(ergotropy)
    collisions_no_env.append(collision)

    one_state = tensor(battery, fresh_ancilla())

#plt.plot(collisions_no_env, ergo_val_no_env, label="No Environment", linewidth=2, linestyle='--')

# Final adjustments
plt.xlabel("Number of Collisions", fontsize=18)
plt.ylabel(" Ergotropy", fontsize=18)
plt.title("Ergotropy for different Partial Swap probabilities", fontsize=18)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

end = time.time()
print("Time Ellapsed:", end - start)
print("In minutes:", (end - start) / 60)