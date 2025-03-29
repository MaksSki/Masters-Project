
import numpy as np
import matplotlib.pyplot as plt
from qutip import *


collision_num = 1000     # Total number of collisions
g = 1                   # Interaction strength
N = 200                 # Number of battery energy levels (levels 0,...,N)
theta = np.pi / 4       # Interaction parameter controlling energy transfer
q = 0.75                # Probability for the qubit to be in the ground state (|0>)
c = 1                   # Coherence parameter



unplug_threshold = 0.85 * N  


battery_int_1 = basis(N+1, 0)
battery1_state = ket2dm(battery_int_1)
battery_2 = ket2dm(basis(N+1, 0))
first_ancilla = basis(2, 1)


A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()


V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()


one_state = ket2dm(tensor(battery_int_1, first_ancilla))


avg_energy_battery1 = []  # For Battery 1
avg_energy_battery2 = []  # For Battery 2
collisions = []           # Collision counter
levels = np.arange(N+1)   # Array of energy levels (0, 1, 2, ..., N)

# Flag to indicate if Battery 1 is unplugged 
battery1_unplugged = False


for collision in range(1, collision_num+1):
    U_dagg = U.dag()
    
    if not battery1_unplugged:
        
        # Battery 1 interacts with its ancilla via the unitary evolution.
        one_state = U * one_state * U_dagg
        battery1_state = one_state.ptrace(0)  # Extract Battery 1 state
        avg1 = np.dot(levels, battery1_state.diag())
        
        # Check if Battery 1 has reached its peak average energy.
        if avg1 >= unplug_threshold:
            battery1_unplugged = True
            print(f"Battery 1 unplugged at collision {collision}")
        
        # Prepare a fresh ancilla state for Battery 1.
        rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
        coherence_term = c * np.sqrt(q * (1 - q)) * (
            basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
        )
        rho_Q = rho_Q + coherence_term
        rho_Q = rho_Q / rho_Q.tr()  # Normalize
        
        # Extract the ancilla from Battery 1
        leftover_ancilla = one_state.ptrace(1)
        
        # Reset Battery 1 
        one_state = tensor(battery1_state, rho_Q)
        
        
        two_state = tensor(battery_2, leftover_ancilla)
        two_state = U * two_state * U_dagg
        battery_2 = two_state.ptrace(0)
        avg2 = np.dot(levels, battery_2.diag())
        
    else:
        
        # Battery 1 no longer interacts and remains unchanged.
        avg1 = np.dot(levels, battery1_state.diag())
        
        
        rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
        coherence_term = c * np.sqrt(q * (1 - q)) * (
            basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
        )
        rho_Q = rho_Q + coherence_term
        rho_Q = rho_Q / rho_Q.tr()  # Normalize
        
        two_state = tensor(battery_2, rho_Q)
        two_state = U * two_state * U_dagg
        battery_2 = two_state.ptrace(0)
        avg2 = np.dot(levels, battery_2.diag())
    
    # Record the average energies for this collision.
    avg_energy_battery1.append(avg1)
    avg_energy_battery2.append(avg2)
    collisions.append(collision)


plt.figure(figsize=(10, 6))
plt.plot(collisions, avg_energy_battery1, label="Battery 1", linewidth=2)
plt.plot(collisions, avg_energy_battery2, label="Battery 2", linewidth=2)
plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Average Energy Level", fontsize=14)
plt.title("Average Energy vs. Number of Collisions", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
