import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
collision_num = 2000   # Total number of collisions
record_step = 1         # Measure average energy every n collisions 
g = 1                  # Interaction strength
theta = np.pi / 4      # Interaction parameter controlling energy transfer
q = 0.75               # Probability of qubit being in excited state
kappa = 1              # Coupling constant
delta_t = 1            # Time step

N_values = np.linspace(50, 1000, num=20, dtype=int)  # Varying N from 20 to 2000

plt.figure(figsize=(8, 5))

for N in N_values:
    battery_int = basis(N+1, 0)  # |0> in battery Hilbert space
    
    # Battery ladder operators
    A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))  # Lowering operator
    A_dag = A.dag()
    
    # Interaction Hamiltonian Battery-Charger
    V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U = (-1j * theta * V).expm()  # Unitary evolution operator
    
    # Battery-Environment Interaction
    sigma_up = sigmap()
    sigma_down = sigmam()
    V_env = np.sqrt(kappa / delta_t) * (tensor(sigma_up, sigma_down) + tensor(sigma_down, sigma_up))
    U_env = (-1j * theta * V_env).expm()
    
    # Initial state
    state = ket2dm(tensor(battery_int, basis(2, 1)))
    
    collision_numbers = []
    fraction_charged_list = []
    
    # Run the collision model simulation
    for collision in range(1, collision_num + 1):
        state = U * state * U.dag()
        
        # Prepare a new qubit in a mixed state with coherence
        c = 1  # Coherence constant
        rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
        coherence_term = c * np.sqrt(q * (1 - q)) * (basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag())
        rho_Q = rho_Q + coherence_term
        rho_Q = rho_Q / rho_Q.tr()
        
        # Reset the qubit while keeping the battery state
        battery_state = state.ptrace(0)
        state = tensor(battery_state, rho_Q)
        
        if collision % record_step == 0:
            battery_state = state.ptrace(0)
            energy_distribution = np.real(battery_state.diag())
            average_energy = sum(n * prob for n, prob in enumerate(energy_distribution))
            collision_numbers.append(collision)
            fraction_charged_list.append(average_energy )
    
    # Plot fraction charged vs collision number
    plt.plot(collision_numbers, fraction_charged_list, label=f"N={N}")

plt.xlabel("Collision Number")
plt.ylabel("Fraction Charged (Mean Energy Level / N)")
plt.title("Battery Charging Fraction vs. Number of Collisions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
