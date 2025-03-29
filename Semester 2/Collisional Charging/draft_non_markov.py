# Author: Maksymilian Skiba
# Non-Markovian Collision Model of N Batteries with Memory Effects

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Global Variables
collision_num = 5000  # Total number of collisions
n_batt = 10          # Number of batteries in the chain
N = 200               # Battery Hilbert space dimension (levels 0,...,N)
theta = np.pi / 4     # Interaction parameter controlling energy transfer
q = 0.75              # Probability for the qubit to be in the excited state |1>
c = 1                 # Coherence parameter for the ancilla
g = .8              # Interaction strength
unplug_threshold = 0.85 * N  # Battery unplug threshold

# Memory parameter (0 = Markovian, 1 = Fully Non-Markovian)
alpha = 1  

# Define ladder operators for the battery
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()

# Interaction Hamiltonian
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  # Unitary evolution operator

def fresh_ancilla():
    """
    Generates a fresh ancilla state defined as a mixture with coherence.
    """
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q = rho_Q + coherence_term
    return rho_Q / rho_Q.tr()

# Initialize each battery
batteries = [ket2dm(basis(N+1, 0)) for _ in range(n_batt)]

# Keep track of which batteries are still plugged (active)
plugged = [True] * n_batt

# Containers to record the average energy for each battery at every collision
avg_energy = [[] for _ in range(n_batt)]
collisions = []
levels = np.arange(N+1)  # Energy levels 0, 1, ..., N

# Initialize first ancilla with a fresh state
previous_ancilla = fresh_ancilla()

def fresh_ancilla_with_memory():
    """
    Generates an ancilla with memory by mixing it with the previous ancilla state.
    """
    global previous_ancilla
    new_ancilla = fresh_ancilla()
    correlated_ancilla = (1 - alpha) * new_ancilla + alpha * previous_ancilla
    previous_ancilla = correlated_ancilla  # Store for next round
    return correlated_ancilla / correlated_ancilla.tr()

# Main Collision Loop
for collision in range(1, collision_num + 1):
    collisions.append(collision)

    # Find the first plugged battery
    first_plugged_index = next((i for i, p in enumerate(plugged) if p), None)

    if first_plugged_index is None:
        break  # Stop if no batteries are left

    # Start with an ancilla that has memory
    current_ancilla = fresh_ancilla_with_memory()

    # Pass the ancilla through the remaining plugged batteries
    for i in range(first_plugged_index, n_batt):
        if plugged[i]:
            # Battery i interacts with the current ancilla
            combined_state = tensor(batteries[i], current_ancilla)
            evolved_state = U * combined_state * U.dag()
            new_battery_state = evolved_state.ptrace(0)
            batteries[i] = new_battery_state
            
            # Compute average energy
            avg_E = np.dot(levels, new_battery_state.diag())

            # Unplug battery if threshold is reached
            if avg_E >= unplug_threshold:
                plugged[i] = False
                print(f"Battery {i+1} unplugged at collision {collision}")

            # Pass the ancilla forward for the next battery
            current_ancilla = evolved_state.ptrace(1)

    # Record the average energy for visualization
    for i in range(n_batt):
        avg_E = np.dot(levels, batteries[i].diag())
        avg_energy[i].append(avg_E)

# Trim collisions to match available energy records
min_length = min(len(e) for e in avg_energy)  # Find the shortest energy record
trimmed_collisions = collisions[:min_length]  # Trim collisions to match

# Plot results safely
plt.figure(figsize=(10, 6))
for i in range(n_batt):
    plt.plot(trimmed_collisions, avg_energy[i][:min_length], label=f"Battery {i+1}", linewidth=2)

plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Average Energy Level", fontsize=14)
plt.title("Non-Markovian Average Energy vs. Collisions for a Chain of Batteries", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
