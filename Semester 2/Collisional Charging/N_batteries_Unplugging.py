#Author: Maksymilian Skiba
# Collision model of N batteries with unplugging at thresholds
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

#Global Variables
collision_num = 2000   # Total number of collisions
n_batt = 3                # Number of batteries in the chain
N = 200                   # Battery Hilbert space dimension (levels 0,...,N)
theta = np.pi / 4         # Interaction parameter controlling energy transfer
q = 0.75                  # Probability for the qubit to be in the excired state |1>
c = 1                     # Coherence parameter for the ancilla
g = 1                     # Interaction strength

# Battery unplug threshold
unplug_threshold = 0.85 * N  


A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()

# Interaction Hamiltonian
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  # Unitary


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


for collision in range(1, collision_num + 1):
    collisions.append(collision)
    
    # Start each collision with a fresh ancilla for battery 1.
    current_ancilla = fresh_ancilla()
    
    # Propagate the ancilla sequentially through the battery chain.
    for i in range(n_batt):
        if plugged[i]:
            # Battery i interacts with the current ancilla.
            combined_state = tensor(batteries[i], current_ancilla)
            evolved_state = U * combined_state * U.dag()
            new_battery_state = evolved_state.ptrace(0)
            avg_E = np.dot(levels, new_battery_state.diag())
            batteries[i] = new_battery_state
            
            # If the battery reaches (or exceeds) 80% charge, unplug it.
            if avg_E >= unplug_threshold:
                plugged[i] = False
                print(f"Battery {i+1} unplugged at collision {collision}")
                # Reset ancilla so subsequent batteries get a fresh one.
                current_ancilla = fresh_ancilla()
            else:
                # Otherwise, pass on the leftover (modified) ancilla.
                current_ancilla = evolved_state.ptrace(1)
        else:
            # Battery i is unplugged so it does not interact.
            # Use a fresh ancilla for the next battery.
            current_ancilla = fresh_ancilla()
    
    # Record the average energy 
    for i in range(n_batt):
        avg_E = np.dot(levels, batteries[i].diag())
        avg_energy[i].append(avg_E)

plt.rcParams["font.family"]="Times New Roman"
plt.tick_params(axis='both',which='major',labelsize=18)
plt.figure(figsize=(10, 6))
for i in range(n_batt):
    plt.plot(collisions, avg_energy[i], label=f"Battery {i+1}", linewidth=2)
plt.xlabel("Number of Collisions", fontsize=18)
plt.ylabel("Average Energy Level", fontsize=18)
plt.title("Average Energy vs. Collisions for a Chain of Batteries", fontsize=18)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
