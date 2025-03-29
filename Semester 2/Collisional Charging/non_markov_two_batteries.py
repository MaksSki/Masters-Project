#Author : Maksymilian Skiba
# Trickle charging of 2 batteries
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
q = .75 # <<<<< Probability of qubit being in ground state (0 means always excited, 1 means always ground)
c=1
# Initial battery state (empty battery)
battery_int_1 = basis(N+1, 0)  # |0> in battery Hilbert space
battery_2 = ket2dm(basis(N+1, 0))
first_ancilla= basis(2,1)
# Define the battery ladder operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))  # Lowering operator
A_dag = A.dag()  # Raising operator

# Define the interaction Hamiltonian following the paper
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))  # Swap interaction

# Unitary evolution operator
U = (-1j * theta * V).expm()

# Initial battery density matrix
one_state = ket2dm(tensor(battery_int_1, first_ancilla))  # Start with an excited qubit

for collision in range(1,collision_num+1):
    U_dagg=U.dag()
    one_state=U*one_state*U_dagg
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))

    # Adding coherence term correctly
    coherence_term = c * np.sqrt(q * (1 - q)) * (basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag())

    rho_Q = rho_Q + coherence_term  # Apply coherence
    rho_Q = rho_Q / rho_Q.tr()  # Ensure proper normalization

    # Re-tensor with the fresh qubit
    ancilla=one_state.ptrace(1)
    one_state = tensor(one_state.ptrace(0), rho_Q)  # Keep battery state and reset qubit
    two_state= (tensor((battery_2),(ancilla)))
    two_state=U*two_state*U_dagg
    battery_1=one_state.ptrace(0)
    battery_2=two_state.ptrace(0)
    # Plot the Energy Distribution
    if collision == observe_after:
        battery_1=one_state.ptrace(0)
        battery_2=two_state.ptrace(0)
        energy_distribution_1 = battery_1.diag()  # Extract probability of each energy level
        energy_distribution_2 = battery_2.diag()
        plt.figure(figsize=(8, 5))
        plt.plot(range(N+1), energy_distribution_1,label="Battery 1")      
        plt.plot(range(N+1), energy_distribution_2,label="Battery 2")
        plt.xlabel("Energy Level")
        plt.ylabel("Probability")
        plt.title(f"Battery Energy Distribution After {observe_after} Collisions")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend()
        plt.show()
        