# Author: Maksymilian Skiba
# Collision model of N batteries with unplugging at thresholds
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Global Variables
collision_num = 1000  # Total number of collisions
n_batt = 3            # Number of batteries in the chain
N = 200                # Battery Hilbert space dimension (levels 0,...,N)
theta = np.pi / 4      # Interaction parameter controlling energy transfer
q = 0.75               # Probability for the qubit to be in the excited state |1>
c = 1                  # Coherence parameter for the ancilla
g = 0.9                # Interaction strength

# Battery unplug threshold (85% of full charge)
unplug_threshold =  0.85*N

# Operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()

# Interaction Hamiltonian
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  # Unitaryå

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

def get_passive_energy(rho, levels):
    """
    Computes the passive state's energy of a battery given its density matrix (rho)
    and the energy levels (assumed in ascending order).

    Steps:
      1. Obtain the eigenvalues (populations) of rho.
      2. Sort them in descending order.
      3. Compute the dot product with the energy levels.
    """
    eigvals, _ = rho.eigenstates()
    eigvals = np.array(eigvals, dtype=float)
    sorted_populations = np.sort(eigvals)[::-1]  # descending order
    passive_energy = np.dot(sorted_populations, levels)
    return passive_energy

# Initialize each battery in the ground state.
batteries = [ket2dm(basis(N+1, 0)) for _ in range(n_batt)]

# Keep track of which batteries are still plugged (active)
plugged = [True] * n_batt

# Containers to record the average energy and ergotropy for each battery at every collision
avg_energy = [[] for _ in range(n_batt)]
ergotropy = [[] for _ in range(n_batt)]
collisions = []
levels = np.arange(N+1)  # Energy levels 0, 1, ..., N

# Run the simulation
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
            
            # If the battery reaches (or exceeds) the unplug threshold, unplug it.
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
    
    # Record the average energy and ergotropy for each battery
    for i in range(n_batt):
        avg_E = np.dot(levels, batteries[i].diag())
        avg_energy[i].append(avg_E)
        
        passive_E = get_passive_energy(batteries[i], levels)
        # Updated ergotropy: difference between average energy and passive energy
        ergotropy[i].append(avg_E - passive_E)

# Plot the results: average energy as a solid line and ergotropy as a dotted line
plt.figure(figsize=(10, 9))
for i in range(n_batt):
    plt.plot(collisions, avg_energy[i], label=f"Battery {i+1} Avg Energy", linewidth=2)
    plt.plot(collisions, ergotropy[i], linestyle=':', label=f"Battery {i+1} Ergotropy", linewidth=2)


# Compute max ergotropy for battery 1
max_erg = max(ergotropy[0])

# Plot the results: average energy, ergotropy, and max‐ergotropy line
# find the collision index of max ergotropy
max_idx       = np.argmax(ergotropy[0])      # zero‐based index into your list
collision_max = collisions[max_idx]          # the collision number

# Plot the results: average energy, ergotropy, and max‐ergotropy marker
plt.figure(figsize=(10, 9))
plt.plot(collisions, avg_energy[0], label="Average Energy", linewidth=2)
plt.plot(collisions, ergotropy[0],  linestyle=':',  label="Ergotropy", linewidth=2)

# vertical line at the collision of max ergotropy
plt.rcParams["font.family"]="Times New Roman"
plt.tick_params(axis='both',which='major',labelsize=18)
plt.axvline(x=collision_max, linestyle='--', label=fr"Peak Ergotropy  $",color='red')

plt.xlabel("Number of Collisions", fontsize=18)
plt.ylabel("Average Energy and Ergotropy",    fontsize=18)
plt.title("Possible Work Extraction for a 3-Level Battery system", fontsize=18)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
# ... (previous code unchanged)

# Plot the results: average energy, ergotropy for all batteries, and max‐ergotropy line for battery 1
plt.figure(figsize=(10, 9))

for i in range(n_batt):
    plt.plot(collisions, avg_energy[i],
             label=f"Battery {i+1} Average Energy",
             linewidth=2)
    plt.plot(collisions, ergotropy[i],
             linestyle=':',
             label=f"Battery {i+1} Ergotropy",
             linewidth=2)

# vertical line at the collision of max ergotropy (for battery 1)
plt.rcParams["font.family"] = "Times New Roman"
plt.tick_params(axis='both', which='major', labelsize=18)
"""
plt.axvline(x=collision_max,
            linestyle='--',
            label="Battery 1 - Peak Ergotropy",
            color='red')
"""""
plt.xlabel("Number of Collisions", fontsize=18)
plt.ylabel("Average Energy and Ergotropy", fontsize=18)
plt.title("Possible Work Extraction for a 3-Level Battery system", fontsize=18)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
