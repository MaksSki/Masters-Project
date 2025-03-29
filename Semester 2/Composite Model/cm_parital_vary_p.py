import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# Parameters
collision_num = 1000  
record_step = 1      
g = 1               
N = 200              
theta = np.pi / 4    
q = 0.75             
kappa = 2 
delta_t = 1          
c = 1                
q_env = 0.5          
omega0=1

# Define functions
def fresh_ancilla():
    """Charger ancilla state (mixture with coherence)."""
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q = rho_Q + coherence_term
    return rho_Q / rho_Q.tr()

def fresh_ancilla_env():
    return ket2dm(basis(2,0))

def partial_swap_map(rho, p):
    """Partial swap quantum map."""
    swap_op = swap(2,2)  # Swap operator
    rho_new = (1 - p) * rho + p * swap_op * rho * swap_op.dag()
    return rho_new / rho_new.tr()

# Battery ladder operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))  
A_dag = A.dag()  

# Interaction Unitaries
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  
V_env = np.sqrt(kappa / delta_t) * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U_env = (-1j * V_env).expm()

# Loop over different p_swap values
plt.figure(figsize=(10, 6))
for p_swap in np.linspace(0, 1, 11):  # p from 0 to 1 in steps of 0.1
    battery_int = basis(N+1, 0)
    ancilla = basis(2, 1)  
    new_env_ancilla = fresh_ancilla_env()
    one_state = ket2dm(tensor(battery_int, ancilla))
    ergotropy_val=[]
    avg_energy_battery = []
    collisions = []
    levels = np.arange(N+1)

    for collision in range(1, collision_num + 1):
        one_state = U * one_state * U.dag()
        battery = one_state.ptrace(0)
        
        one_state_env = tensor(battery, new_env_ancilla)
        one_state_env = U_env * one_state_env * U_env.dag()
        
        battery = one_state_env.ptrace(0)
        updated_env_ancilla = one_state_env.ptrace(1)
        
        combined = tensor(fresh_ancilla_env(), updated_env_ancilla)
        swapped = partial_swap_map(combined, p_swap)
        new_env_ancilla = swapped.ptrace(0)
        
        one_state = tensor(battery, new_env_ancilla)
        
        # Define the battery Hamiltonian (energy levels as basis states)
        H_B = sum(n * ket2dm(basis(N+1, n)) for n in range(N+1))

# Compute battery energy as the expectation value of H_B
        energy = expect(H_B, battery)

# Compute ergotropy: Maximum extractable energy via unitary operations
        eigenvalues, eigenstates = H_B.eigenstates()
        passive_state = sum(eigenvalues[i] * ket2dm(eigenstates[i]) for i in range(N+1))
        ergotropy = energy - expect(H_B, passive_state)  # Ergotropy = Energy - Passive Energy

        avg_energy_battery.append(energy)
        ergotropy_val.append(ergotropy)
        collisions.append(collision)
    
    plt.plot(collisions, avg_energy_battery, label=f"p={p_swap:.1f}", linewidth=2)
    plt.plot(collisions, ergotropy_val, label=f"p={p_swap:.1f}", linewidth=2)
    

# Plot settings
plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Average Energy Level", fontsize=14)
plt.title("Average Energy Level vs. Number of Collisions", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
