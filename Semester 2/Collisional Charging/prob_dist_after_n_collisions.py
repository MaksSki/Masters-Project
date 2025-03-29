import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
collision_num = 500  # Total number of collisions
g = 1  # Interaction strength
N = 200  # Number of battery levels
dt = 0.1  # Time step
theta = np.pi / 4  # Interaction parameter controlling energy transfer
observe_after = collision_num# <<<<< Change this to observe energy distribution at a specific collision step
q = 1 # <<<<< Probability of qubit being in ground state (0 means always excited, 1 means always ground)

# Initial battery state (empty battery)
battery_int = basis(N+1, 0)  # |0> in battery Hilbert space

# Define the battery ladder operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))  # Lowering operator
A_dag = A.dag()  # Raising operator

# Define the interaction Hamiltonian following the paper
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))  # Swap interaction

# Unitary evolution operator
U = (-1j * theta * V).expm()

# Initial battery density matrix
state = ket2dm(tensor(battery_int, basis(2, 1)))  # Start with an excited qubit

# Simulate Collisions
for collision in range(1, collision_num + 1):
    # Apply unitary evolution
    U_dagg = U.dag()
    state = U * state * U_dagg

    # Prepare a new qubit in a mixed state (Fixed version)
    c = 1 #  coherence

    # Assign proper populations to ground and excited states
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))

    # Adding coherence term correctly
    coherence_term = c * np.sqrt(q * (1 - q)) * (basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag())

    rho_Q = rho_Q + coherence_term  # Apply coherence
    rho_Q = rho_Q / rho_Q.tr()  # Ensure proper normalization

    # Re-tensor with the fresh qubit
    state = tensor(state.ptrace(0), rho_Q)  # Keep battery state and reset qubit

    # Plot at the specified observation point
    if collision == observe_after:
        battery_state = state.ptrace(0)  # Partial trace over qubit to get battery state
        energy_distribution = battery_state.diag()  # Extract probability of each energy level

        # Plot the energy distribution
        plt.figure(figsize=(8, 5))
        plt.bar(range(N+1), energy_distribution, alpha=0.7)
        plt.plot(range(N+1), energy_distribution)
        plt.xlabel("Energy Level")
        plt.ylabel("Probability")
        plt.title(f"Battery Energy Distribution After {observe_after} Collisions")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()
