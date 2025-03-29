import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Global Variables
collision_num = 4000      # Total number of collisions
n_batt = 3                # Number of batteries in the chain
N = 200                   # Battery Hilbert space dimension (levels 0,...,N)
theta = np.pi / 4         # Interaction parameter controlling energy transfer
q_range = np.linspace(0,1,11)  # Probability for the qubit to be in the excited state |1>
c = 1                     # Coherence parameter for the ancilla
g_range = np.linspace(0.1, 2, 20)  # Interaction strength

# Battery unplug threshold
unplug_threshold = 0.85 * N  

# Define ladder operator for the battery
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()

def fresh_ancilla(q):
    """
    Create and return a fresh ancilla state for a given excitation probability q.
    """
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q += coherence_term
    return rho_Q
 
# Create a matrix to record the final collision number for the last battery
# Rows correspond to g_range, columns to q_range.
final_coll_matrix = np.zeros((len(g_range), len(q_range)))

# Loop over interaction strength (g) and qubit excitation probability (q)
for i_g, g in enumerate(g_range):
    # Define the interaction Hamiltonian and corresponding unitary evolution for the current g
    V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U = (-1j * theta * V).expm()
    
    for i_q, q in enumerate(q_range):
        # Initialize the battery chain: each battery starts in its ground state.
        batteries = [ket2dm(basis(N+1, 0)) for _ in range(n_batt)]
        plugged = [True] * n_batt  # True if battery is still connected.
        
        final_collision = collision_num  # Default value if the battery never unplugs.
        
        # Run the collision process.
        for collision in range(1, collision_num + 1):
            # Get a fresh ancilla for this collision (based on the current q)
            current_ancilla = fresh_ancilla(q)
            
            # Propagate the ancilla sequentially through the battery chain.
            for i in range(n_batt):
                if plugged[i]:
                    combined_state = tensor(batteries[i], current_ancilla)
                    evolved_state = U * combined_state * U.dag()
                    new_battery_state = evolved_state.ptrace(0)
                    batteries[i] = new_battery_state
                    avg_E = np.dot(np.arange(N+1), new_battery_state.diag())
                    
                    # Check if the battery's average energy crosses the unplug threshold.
                    if avg_E >= unplug_threshold:
                        plugged[i] = False
                        # If this is the last battery, record the collision and exit.
                        if i == n_batt - 1:
                            final_collision = collision
                            break
                        # Reset the ancilla if a battery in the chain unplugs.
                        current_ancilla = fresh_ancilla(q)
                    else:
                        # Propagate the (evolved) ancilla to the next battery.
                        current_ancilla = evolved_state.ptrace(1)
                else:
                    # If battery already unplugged, use a fresh ancilla.
                    current_ancilla = fresh_ancilla(q)
            # If the last battery has been unplugged, break out of the collision loop.
            if not plugged[-1]:
                break
        
        final_coll_matrix[i_g, i_q] = final_collision

# Create a contour plot: x-axis = g, y-axis = q, color = final collision number.
G, Q = np.meshgrid(g_range, q_range, indexing='ij')

plt.figure(figsize=(8, 6))
cp = plt.contourf(G, Q, final_coll_matrix, levels=25, cmap='viridis')
plt.colorbar(cp, label='Collision Number for Last Battery Unplug')
plt.xlabel('Interaction Strength (g)', fontsize=14)
plt.ylabel('Qubit Excitation Probability (q)', fontsize=14)
plt.title(('Non-Markovian Collisions to a Full Charge for ',n_batt, 'Batteries'), fontsize=16)
plt.tight_layout()
plt.show()
