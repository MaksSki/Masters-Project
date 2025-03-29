# Author : Maksymilian Skiba
# Modelling a quantum battery interacting with qubits via collisions
# Probabilty of battery being full charged after each collision

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
collision_num = 700 # Number of collisions
g = 1  # Interaction strength
N = 200  # Number of battery levels
dt = 0.1  # Time step
theta = np.pi / 4  # Interaction parameter controlling energy transfer

# Initial battery state (empty battery)
battery_int = basis(N+1, 0)  # |0> in battery Hilbert space

# Define the battery ladder operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))  # Lowering operatorßß
A_dag = A.dag()  # Raising operator

# Define the interaction Hamiltonian following the paper
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))  # Swap interaction

# Corrected Unitary evolution operator
U = (-1j * theta * V).expm()

# Store battery populations over time
data = []

# Initial battery density matrix
state = ket2dm(tensor(battery_int, basis(2, 1)))  # Start with |e> qubit

for _ in range(collision_num):
    # Apply unitary evolution
    U_dagg = U.dag()
    state = U * state * U_dagg

    # Prepare a new qubit in a **mixed** state based on q=0.5
    q = 0.75  # Prbability of ground state
    c = 1.0  # Max coherence
    alpha = 0  # Phase angleß
    
    rho_Q = q * ket2dm(basis(2, 0)) + (1 - q) * ket2dm(basis(2, 1))  # Mixed state
    rho_Q += c * np.sqrt(q * (1 - q)) * (basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag())  # Coherence term

    # Re-tensor with the fresh qubit
    state = tensor(state.ptrace(0), rho_Q)  # Only keep battery state and reset qubit

    # Trace out the qubit to get the battery state
    battery_state = state.ptrace(0)  # Partial trace over qubit
    data.append(battery_state.diag())  # Store energy distribution

# Convert history to numpy array for plotting
data = np.array(data)

# Plot the energy distribution of the battery at each step
# Extract probability of final energy level (|N> state) at each step
final_level_prob = data[:, -1]  # Last column corresponds to |N> level

# Plot probability of being in the final energy level over collisions
plt.plot(range(1, collision_num + 1), final_level_prob, marker='', linestyle='-')

plt.xlabel("Number of Collisions")
plt.ylabel("Probability of Final Energy Level")
plt.title("Probability of Battery Reaching Maximum Energy Over Time")
plt.grid()
plt.show()
