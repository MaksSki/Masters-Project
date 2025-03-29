import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
collision_num = 800     # Total number of collisions
record_step = 1         # Measure average energy every n collisions 
g = 1                    # Interaction strength
N = 200                 # Number of battery levels
theta = np.pi / 4        # Interaction parameter controlling energy transfer
q = .75               # Probability of qubit being in excited state (0 means always excited, 1 means always ground)
kappa = 1  # coupling constant
delta_t = 1  # time step
# Initial battery state (empty battery)
battery_int = basis(N+1, 0)  # |0> in battery Hilbert space
### BATTERY - CHARGER
# Define the battery ladder operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))  # Lowering operator
A_dag = A.dag()  # Raising operator

# Define the interaction Hamiltonian Battery-Charge (swap interaction)
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  # Unitary evolution operator
### BATTERY - ENVIROMENT
sigma_up=sigmap()
sigma_down=sigmam()
I_matrix=qeye(2)
V_env = np.sqrt(kappa / delta_t) * (tensor(sigma_up, sigma_down) + tensor(sigma_down, sigma_up))
U_env = (-1j * theta * V_env).expm()
###





# Initial total state: battery in state |0> and qubit in excited state |1>
state = ket2dm(tensor(battery_int, basis(2, 1)))

# Lists to store collision numbers and corresponding average energies
collision_numbers = []
avg_energy_list = []

# Run the collision model simulation
for collision in range(1, collision_num + 1):
    # Apply unitary evolution (interaction between battery and qubit)
    state = U * state * U.dag()
    
    # Prepare a new qubit in a mixed state with coherence
    c = 1  # Coherence constant
    # Assign populations: (1-q) for excited, q for ground
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    # Add the coherence term
    coherence_term = c * np.sqrt(q * (1 - q)) * (basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag())
    rho_Q = rho_Q + coherence_term
    rho_Q = rho_Q / rho_Q.tr()  # Ensure proper normalization
    
    # Reset the qubit while keeping the battery state
    battery_state = state.ptrace(0)
    state = tensor(battery_state, rho_Q)
    
    # Record average energy at the specified collision step
    if collision % record_step == 0:
        battery_state = state.ptrace(0)
        energy_distribution = np.real(battery_state.diag())  # Extract diagonal probabilities
        # Compute the average energy using the level index as the energy value
        average_energy = sum(n * prob for n, prob in enumerate(energy_distribution))
        collision_numbers.append(collision)
        avg_energy_list.append(average_energy)

# Plot average energy vs. collision number
plt.figure(figsize=(8, 5))
plt.plot(collision_numbers, avg_energy_list, marker='-', linestyle='-', markersize=4)
plt.xlabel("Collision Number")
plt.ylabel("Average Energy Level")
plt.title("Battery Average Energy vs. Number of Collisions")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
print(f"Max average energy {max(avg_energy_list)} reached at collision step {collision_numbers[np.argmax(avg_energy_list)]}")

