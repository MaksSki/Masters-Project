import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# Parameters
collision_num = 1000  # Total number of collisions
record_step = 1      # Measure average energy every n collisions
g = 1                # Interaction strength
N = 200              # Number of battery levels
theta = np.pi / 4    # Interaction parameter controlling energy transfer
q = 0.75             # Probability of qubit being in excited state (0 means always excited, 1 means always ground)
delta_t = 1          # Time step
c = 1                # Coherence term scaling factor
q_env = 0.5          # Probability of environmental qubit being in excited state
p=1
# Define the fresh ancilla functions
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
def get_passive_energy(rho, levels):
    """
    Computes the passive state's energy of a battery given its density matrix (rho)
    and the energy levels (assumed in ascending order).

    Steps:
      1. Obtain the eigenvalues (populations) of rho.
      2. Sort them in descending order.
      3. Compute the dot product with the energy levels.
    """
    eigvals, _ = rho.eigenstates()
    eigvals = np.array(eigvals, dtype=float)
    sorted_populations = np.sort(eigvals)[::-1]  # descending order
    passive_energy = np.dot(sorted_populations, levels)
    return passive_energy

# Initial battery state (empty battery)
battery_int = basis(N+1, 0)  # |0> in battery Hilbert space

# Define the battery ladder operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))  # Lowering operator
A_dag = A.dag()  # Raising operator

# Define the interaction Hamiltonian Battery-Charge (swap interaction)
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  # Unitary evolution operator
U_dag = U.dag()

# Collision order: Battery - Charger, Battery - Environment
ancilla = basis(2, 1)  # Charger ancilla
env_ancilla = ket2dm(basis(2, 0))  # Environment ancilla

# Loop over different values of kappa
kappa_values = [0.001,0.25,0.5,1.25,(5/3), 2.5, 5 ,10]  # Example values for kappa
plt.figure(figsize=(10, 6))

for kappa in kappa_values:
    # Define the interaction Hamiltonian Battery-Environment (swap interaction) for each kappa
    V_env = np.sqrt(kappa / delta_t) * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U_env = (-1j * 1 * V_env).expm()
    U_env=np.sqrt(p)*U_env +np.sqrt(1-p)*tensor(qeye(N+1),qeye(2))
   # U_env= partial_swap_operator(p,N+1)
    U_env_dag = U_env.dag()

    # Initial state for the composite system
    one_state = ket2dm(tensor(battery_int, ancilla))  # Initial state of the composite system

    avg_energy_battery = []
    collisions = []
    levels = np.arange(N+1)

    for collision in range(1, collision_num + 1):
        # Battery-Charger interaction
        one_state = U * one_state * U_dag  # Apply unitary evolution for the battery-charger interaction
        carried_ancilla = one_state.ptrace(1)  # Take the partial trace over the charger
        battery = one_state.ptrace(0)  # Take the partial trace over the battery
        probability_no_env = battery.diag()
        average_no_env = np.dot(levels, probability_no_env)
        
        # Now, interact only the battery with the environment
        one_state_env = tensor(battery, env_ancilla)  # Apply the environment to the state (battery and environment)
        one_state_env = U_env * one_state_env * U_env_dag  # Apply unitary evolution for the environment
        
        # After the environment interaction, extract the battery's state
        battery = one_state_env.ptrace(0)  # Get the battery's state after the environment interaction
        
        # Calculate average energy of the battery
        probability = battery.diag()  # Diagonal elements of the density matrix (probabilities)
        average = np.dot(levels, probability)
        total=average-get_passive_energy(battery,levels)
        avg_energy_battery.append(total)
        collisions.append(collision)
        
        # Reset the environment state after each collision
        one_state = tensor(battery, fresh_ancilla())  # Reset the environment state

    # Plot the average energy of the battery for the current kappa
    plt.plot(collisions, avg_energy_battery, label=f"$g/kappa = {g/kappa}$", linewidth=2)

# Plot customization
plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Ergotropy", fontsize=14)
plt.title("Ergotropy vs. Number of Collisions for Different $g/kappa$", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
