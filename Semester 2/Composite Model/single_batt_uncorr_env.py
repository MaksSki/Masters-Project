# Author : Maksymilian
# In Hilbert Space modelling a composite collisional model for a simple quantum battery as a MPS and uncomputing the interacted enviroment.
import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# Parameters
collision_num = 1000  # Total number of collisions
record_step = 1      # Measure average energy every n collisions
g = 1               # Interaction strength
N = 200              # Number of battery levels
theta = np.pi / 4    # Interaction parameter controlling energy transfer
q = 0.75             # Probability of qubit being in excited state (0 means always excited, 1 means always ground)
kappa = 4 # Coupling constant
delta_t = 1          # Time step
c = 1                # Coherence term scaling factor
q_env = 0.5          # Probability of environmental qubit being in excited state

def fresh_ancilla():
    """
    Generates a fresh ancilla state defined as a mixture with coherence.
    """
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q = rho_Q + coherence_term
    return rho_Q / rho_Q.tr()

def fresh_ancilla_env():
    """
    Generates a fresh ancilla state defined as a mixture with coherence for the environment.
    """
    rho_Q = (1 - q_env) * ket2dm(basis(2, 1)) + q_env * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q_env * (1 - q_env)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q = rho_Q + coherence_term
    return rho_Q / rho_Q.tr()

# Initial battery state (empty battery)
battery_int = basis(N+1, 0)  # |0> in battery Hilbert space

# Define the battery ladder operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))  # Lowering operator
A_dag = A.dag()  # Raising operator

# Define the interaction Hamiltonian Battery-Charge (swap interaction)
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  # Unitary evolution operator
U_dag = U.dag()

# Define the interaction Hamiltonian Battery-Environment (swap interaction)
V_env = np.sqrt(kappa / delta_t) * (tensor(A,sigmap()) + tensor(A_dag, sigmam()))
U_env = (-1j * 1 * V_env).expm()
U_env_dag = U_env.dag()

# Collision order: Battery - Charger, Battery - Environment
ancilla = basis(2, 1)  # Charger ancilla
env_ancilla = ket2dm(basis(2,0))  # Environment ancilla
one_state = ket2dm(tensor(battery_int, ancilla))  # Initial state of the composite system
avg_energy_no_env=[]
avg_energy_battery = []
collisions = []
levels = np.arange(N+1)

for collision in range(1, collision_num + 1):
    # Battery-Charger interaction
    one_state = U * one_state * U_dag  # Apply unitary evolution for the battery-charger interaction
    carried_ancilla = one_state.ptrace(1)  # Take the partial trace over the charger
    battery = one_state.ptrace(0)  # Take the partial trace over the battery
    probability_no_env=battery.diag()
    average_no_env=np.dot(levels,probability_no_env)
    avg_energy_no_env.append(average_no_env)
    
    # Now, interact only the battery with the environment
    # Create a tensor product of the battery and environment state only
    one_state_env = tensor(battery, env_ancilla)  # Apply the environment to the state (battery and environment)
    
    # Apply the unitary evolution for the environment interaction
    one_state_env = U_env * one_state_env * U_env_dag  # Apply unitary evolution for the environment
    
    # After the environment interaction, extract the battery's state
    battery = one_state_env.ptrace(0)  # Get the battery's state after the environment interaction
    
    # Calculate average energy of the battery
    probability = battery.diag()  # Diagonal elements of the density matrix (probabilities)
    average = np.dot(levels, probability)
    avg_energy_battery.append(average)
    collisions.append(collision)
    
    # Reset the environment state after each collision
    one_state = tensor(battery, fresh_ancilla())  # Reset the environment state

# Plot the average energy of the battery over collisions
plt.figure(figsize=(10, 6))

plt.plot(collisions, avg_energy_battery, label="Battery Energy Levels - Composite", linewidth=2)
plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Average Energy Level", fontsize=14)
plt.title("Average Energy Level vs. Number of Collisions", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
