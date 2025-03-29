import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ----------------------
# Global Parameters
# ----------------------
collision_num = 2000  # Total number of collisions
n_batt = 5            # Number of batteries in the chain
N = 200                # Battery Hilbert space dimension (levels 0,...,N)
theta = np.pi / 4      # Interaction parameter controlling energy transfer
q = 0.75               # Probability for the qubit to be in the excited state |1>
c = 1                  # Coherence parameter for the ancilla
g = .5                  # Interaction strength

# Unplug only if:
#   1) Ergotropy fails to exceed its old maximum for 'look_ahead' consecutive collisions
#   2) Battery's average energy >= 85% of N
look_ahead = 0
energy_threshold = 0.75 * N

# ----------------------
# Ancilla / Battery Setup
# ----------------------
def fresh_ancilla():
    """
    Generates a fresh ancilla state defined as a mixture with coherence.
    """
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q += coherence_term
    return rho_Q / rho_Q.tr()

# Initialize each battery in the ground state.
batteries = [ket2dm(basis(N+1, 0)) for _ in range(n_batt)]

# Keep track of which batteries are still plugged (active)
plugged = [True] * n_batt

# ----------------------
# Hamiltonians / Operators
# ----------------------
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()

V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  # Unitary evolution operator

def get_passive_energy(rho, levels):
    """
    Computes the passive state's energy of a battery given its density matrix (rho)
    and the energy levels (assumed in ascending order).

      1. Obtain the eigenvalues (populations) of rho.
      2. Sort them in descending order.
      3. Compute the dot product with the energy levels.
    """
    eigvals, _ = rho.eigenstates()
    eigvals = np.array(eigvals, dtype=float)
    sorted_populations = np.sort(eigvals)[::-1]  # descending order
    passive_energy = np.dot(sorted_populations, levels)
    return passive_energy

# ----------------------
# Simulation Containers
# ----------------------
avg_energy = [[] for _ in range(n_batt)]
ergotropy = [[] for _ in range(n_batt)]
collisions = list(range(1, collision_num + 1))
levels = np.arange(N+1)  # Energy levels: 0, 1, 2, ..., N

# Keep track of the maximum ergotropy observed so far for each battery
max_erg_so_far = [0.0] * n_batt

# Keep track of how many consecutive collisions each battery remains below its maximum
dip_count = [0] * n_batt

# ----------------------
# Main Collision Loop
# ----------------------
for collision in collisions:
    
    # Start each collision with a fresh ancilla for battery 1.
    current_ancilla = fresh_ancilla()
    
    # Propagate the ancilla sequentially through the battery chain.
    for i in range(n_batt):
        if plugged[i]:
            # 1) Evolve combined state
            combined_state = tensor(batteries[i], current_ancilla)
            evolved_state = U * combined_state * U.dag()
            
            # 2) Trace out ancilla to get new battery state
            new_battery_state = evolved_state.ptrace(0)
            batteries[i] = new_battery_state
            
            # 3) Calculate battery's ergotropy
            avg_E = np.dot(levels, new_battery_state.diag())
            passive_E = get_passive_energy(new_battery_state, levels)
            new_erg = avg_E - passive_E
            
            # 4) Compare to old maximum
            if new_erg > max_erg_so_far[i]:
                # Ergotropy has increased; update maximum
                max_erg_so_far[i] = new_erg
                dip_count[i] = 0
            else:
                # Ergotropy did not exceed the old maximum
                dip_count[i] += 1
                
                # Check both conditions for unplugging:
                #  a) dip_count has reached look_ahead
                #  b) average energy >= 85% of N
                if dip_count[i] >= look_ahead and avg_E >= energy_threshold:
                    plugged[i] = False
                    print(f"Battery {i+1} unplugged at collision {collision}")
                    # Reset ancilla for subsequent batteries
                    current_ancilla = fresh_ancilla()
                    # Stop processing current battery; move to next
                    continue
            
            # 5) Pass on the leftover ancilla to the next battery
            current_ancilla = evolved_state.ptrace(1)
        else:
            # Battery i is unplugged; skip interaction
            # Use a fresh ancilla for the next battery
            current_ancilla = fresh_ancilla()
    
    # Record average energy and ergotropy for each battery after this collision
    for i in range(n_batt):
        avg_E = np.dot(levels, batteries[i].diag())
        passive_E = get_passive_energy(batteries[i], levels)
        ergotropy_val = avg_E - passive_E
        
        avg_energy[i].append(avg_E)
        ergotropy[i].append(ergotropy_val)

# ----------------------
# Plotting Results
# ----------------------
plt.figure(figsize=(10, 6))
for i in range(n_batt):
    plt.plot(collisions, avg_energy[i], label=f"Battery {i+1} Avg Energy", linewidth=2)
    plt.plot(collisions, ergotropy[i], linestyle=':', label=f"Battery {i+1} Ergotropy", linewidth=2)

plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Energy / Ergotropy", fontsize=14)
plt.title("Average Energy and Ergotropy vs. Collisions", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
