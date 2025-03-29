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
kappa = 4 
delta_t = 1          
c = 1                
q_env = 0.5          
p_swap = 0.5  # Partial swap strength (0 = no swap, 1 = full swap)

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
    """Partial swap quantum map from Eq. (14)."""
    
    swap_op = swap(2,2)  # Generate the swap operator for the full system
    
    # Apply the partial swap map: (1 - p) * rho + p * S * rho * S†
    rho_new = (1 - p) * rho + p * swap_op * rho * swap_op.dag()
    
    # Ensure the trace remains 1 (should be unnecessary, but for safety)
    return rho_new / rho_new.tr()


# Initial battery state
battery_int = basis(N+1, 0)

# Battery ladder operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))  
A_dag = A.dag()  

# Interaction Unitaries
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()  

V_env = np.sqrt(kappa / delta_t) * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U_env = (-1j * V_env).expm()

# Initialize system
ancilla = basis(2, 1)  
new_env_ancilla = fresh_ancilla_env()
one_state = ket2dm(tensor(battery_int, ancilla))

avg_energy_no_env = []
avg_energy_battery = []
collisions = []
levels = np.arange(N+1)

for collision in range(1, collision_num + 1):
    # --- Battery-Charger interaction ---
    one_state = U * one_state * U.dag()
    battery = one_state.ptrace(0)
            
    one_state_env = tensor(battery, new_env_ancilla)
    one_state_env = U_env * one_state_env * U_env.dag()
            
    battery = one_state_env.ptrace(0)
    updated_env_ancilla = one_state_env.ptrace(1)
            
            
    combined = tensor(fresh_ancilla_env(), updated_env_ancilla)
    swapped = partial_swap_map(combined, p_swap)
    new_env_ancilla = swapped.ptrace(0)
            
    one_state= tensor(battery,new_env_ancilla)
            
            
    #probability = battery.diag()
    #average_energy = np.dot(levels, probability)
    #ergotropy = average_energy - get_passive_energy(battery, levels)

    # --- Calculate battery energy ---
    probability = battery.diag()
    average = np.dot(levels, probability)
    avg_energy_battery.append(average)
    collisions.append(collision)
    
    # --- Reset state for next loop ---
    one_state = tensor(battery, fresh_ancilla())

# --- Plot results ---
plt.figure(figsize=(10, 6))
plt.plot(collisions, avg_energy_battery, label=("p=", p_swap) ,linewidth=2)
plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Average Energy Level", fontsize=14)
plt.title("Average Energy Level vs. Number of Collisions", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
