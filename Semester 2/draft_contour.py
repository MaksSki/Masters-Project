import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import time 
start=time.time()
# Global Variables
collision_num = 2000      # Total number of collisions
n_batt = 3                # Number of batteries in the chain
N = 200                   # Battery Hilbert space dimension (levels 0,...,N)
theta = np.pi / 4         # Interaction parameter controlling energy transfer
q_range = np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75])  # Qubit excitation probability
c = 1                     # Coherence parameter for the ancilla
g_range = np.linspace(0, 2, 5)  # Interaction strength

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

# Create a matrix to record the final collision number for all batteries unplugging.
# Rows correspond to g_range and columns correspond to q_range.
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
        
        # Run the collision process.
        for collision in range(1, collision_num + 1):
            # Get a fresh ancilla for this collision based on the current q.
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
                        # Reset the ancilla for the next battery.
                        current_ancilla = fresh_ancilla(q)
                    else:
                        # Propagate the (evolved) ancilla to the next battery.
                        current_ancilla = evolved_state.ptrace(1)
                else:
                    # If battery is already unplugged, use a fresh ancilla.
                    current_ancilla = fresh_ancilla(q)
                    
            # After processing all batteries, check if they are all unplugged.
            if all(not status for status in plugged):
                final_coll_matrix[i_g, i_q] = collision
                break
        else:
            # If not all batteries unplugged within collision_num, record collision_num.
            final_coll_matrix[i_g, i_q] = collision_num

# Create a contour plot: x-axis = g, y-axis = q, and color representing final collision number.
G, Q = np.meshgrid(g_range, q_range, indexing='ij')

plt.figure(figsize=(8, 6))
cp = plt.contourf(G, Q, final_coll_matrix, levels=20, cmap='viridis')
plt.colorbar(cp, label='Collision # for All Batteries Unplug')
plt.xlabel('Interaction Strength (g)', fontsize=14)
plt.ylabel('Qubit Excitation Probability (q)', fontsize=14)
plt.title('Final Collision Number vs. g and q', fontsize=16)
plt.tight_layout()
plt.show()
end=time.time()
print('Code Runtime: ',(end-start))