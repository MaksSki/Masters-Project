import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Global Variables
collision_num = 2000   # Total number of collisions
n_batt = 3             # Number of batteries in the chain
N = 200                # Battery Hilbert space dimension (levels 0,...,N)
theta = np.pi / 4      # Interaction parameter controlling energy transfer
q = 0.75               # Probability for the qubit to be in the excited state |1>
c = 1                  # Coherence parameter for the ancilla
g = 1                  # Interaction strength

# Battery unplug threshold
unplug_threshold = .85 * N

# Operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()

# Interaction Hamiltonian
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  # Unitary

# Function to generate a fresh ancilla state defined as a mixture with coherence.
def fresh_ancilla():
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q = rho_Q + coherence_term
    return rho_Q / rho_Q.tr()

# Initialize each battery in the ground state.
batteries = [ket2dm(basis(N+1, 0)) for _ in range(n_batt)]

# Keep track of which batteries are still plugged (active)
plugged = [True] * n_batt

# Containers to record the average energy, ergotropy, and efficiency for each battery at every collision
avg_energy = [[] for _ in range(n_batt)]
ergotropy = [[] for _ in range(n_batt)]
efficiency = [[] for _ in range(n_batt)]
collisions = []
levels = np.arange(N+1)  # Energy levels 0, 1, ..., N

# New function to compute the passive energy of a battery state.
def get_passive_energy(rho, levels):
    """
    Computes the passive state's energy of a battery given its density matrix (rho)
    and the energy levels (assumed in ascending order).
    
    The procedure is:
      1. Diagonalize rho to obtain its eigenvalues.
      2. Sort these eigenvalues in descending order.
      3. Multiply each sorted eigenvalue with the corresponding energy (levels are in ascending order).
      4. Sum the products to obtain the passive energy.
    """
    # Obtain the eigenvalues (populations) of the battery state
    eigvals, _ = rho.eigenstates()
    eigvals = np.array(eigvals, dtype=float)  # ensure they are floats
    # Sort the populations in descending order
    sorted_populations = np.sort(eigvals)[::-1]
    # Compute the passive energy
    passive_energy = np.dot(sorted_populations, levels)
    return passive_energy

# Function to calculate the ergotropy of the ancillary (ancilla) qubit system.
def get_ergotropy_ancilla(q, c):
    EQ = (1/2) * (1 - 2*q + np.sqrt((1 - 2*q)**2 + 4 * c**2 * q * (1 - q)))
    return EQ

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
                current_ancilla = fresh_ancilla()
            else:
                # Otherwise, pass on the leftover (modified) ancilla.
                current_ancilla = evolved_state.ptrace(1)
        else:
            current_ancilla = fresh_ancilla()
    
    # Record the average energy, ergotropy, and efficiency for each battery.
    for i in range(n_batt):
        avg_E = np.dot(levels, batteries[i].diag())
        avg_energy[i].append(avg_E)
        
        # Compute passive energy and then ergotropy.
        passive_E = get_passive_energy(batteries[i], levels)
        ergotropy[i].append(avg_E - passive_E)
        
        E_Q = get_ergotropy_ancilla(q, c)  # Ergotropy of the ancillary system
        efficiency[i].append(ergotropy[i][-1] / (collision * E_Q))  # Efficiency

# Plot the results
plt.figure(figsize=(10, 6))
# Plot efficiency vs collisions for each battery
for i in range(n_batt):
    plt.plot(collisions, efficiency[i], label=f"Battery {i+1} (Efficiency)", linewidth=2)

plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Efficiency", fontsize=14)
plt.title("Efficiency vs. Collisions for a Chain of Batteries", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
