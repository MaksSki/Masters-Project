
import numpy as np
import matplotlib.pyplot as plt
from qutip import *


collision_num = 2500  # Total number of collisions
g = 1               # Interaction strength
N = 200             # Number of battery energy levels (0,...,N)
theta = np.pi / 4   # Interaction parameter controlling energy transfer
q = 0.75            # Probability for the qubit to be in the ground state (|0>)
c = 1               # Coherence parameter


# Initial States

battery_int_1 = basis(N+1, 0)  # pure state for Battery 1

# Battery 2: starts empty as a density matrix.
battery_2 = ket2dm(basis(N+1, 0))

# Fresh ancilla for Battery 1: start with an excited qubit |1>.
first_ancilla = basis(2, 1)


A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()


# Define the Interaction Hamiltonian and Unitary Operator

# V = g (A ⊗ σ⁺ + A† ⊗ σ⁻) implements a swap-like interaction.
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  # Unitary evolution operator

#
# one_state is the density matrix for the combined Battery 1 and ancilla.
one_state = ket2dm(tensor(battery_int_1, first_ancilla))


avg_energy_battery1 = []  # to store average energy Battery 1 after each collision
avg_energy_battery2 = []  # to store average energy Battery 2 after each collision
collisions = []           # collision counter


levels = np.arange(N+1)


for collision in range(1, collision_num+1):
    
    U_dagg = U.dag()
    one_state = U * one_state * U_dagg

    
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q = rho_Q + coherence_term
    rho_Q = rho_Q / rho_Q.tr()  # normalize

    
    leftover_ancilla = one_state.ptrace(1)

    
    battery1_state = one_state.ptrace(0)
    one_state = tensor(battery1_state, rho_Q)

    
    two_state = tensor(battery_2, leftover_ancilla)
    two_state = U * two_state * U_dagg

    
    battery_1 = one_state.ptrace(0)
    battery_2 = two_state.ptrace(0)

    
    p1 = battery_1.diag()
    avg1 = np.dot(levels, p1)
    
    p2 = battery_2.diag()
    avg2 = np.dot(levels, p2)

    avg_energy_battery1.append(avg1)
    avg_energy_battery2.append(avg2)
    collisions.append(collision)


plt.figure(figsize=(10, 6))
plt.plot(collisions, avg_energy_battery1, label="Battery 1", linewidth=2)
plt.plot(collisions, avg_energy_battery2, label="Battery 2", linewidth=2)
plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Average Energy Level", fontsize=14)
plt.title("Average Energy Level vs. Number of Collisions", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
