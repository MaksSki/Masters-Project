import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import time

# Parameters
collision_num = 1000  
record_step = 1      
g = 1               
N = 200              
theta = np.pi / 4    
q = 0.75             
delta_t = 1          
c = 1                
q_env = 0.5          
kappa_values = np.array([0.0001, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4.0])
p_values = [0, 0.2, 0.4, 0.6, 0.8, 1]  # Different partial swap strengths

start = time.time()

def fresh_ancilla():
    """Charger ancilla state (mixture with coherence)."""
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q = rho_Q + coherence_term
    return rho_Q / rho_Q.tr()

def fresh_ancilla_env():
    return ket2dm(basis(2, 0))

def partial_swap_map(rho, p):
    """Partial swap quantum map from Eq. (14)."""
    swap_op = swap(2, 2)  # Generate the swap operator for the full system
    rho_new = (1 - p) * rho + p * swap_op * rho * swap_op.dag()
    return rho_new / rho_new.tr()

def get_passive_energy(rho, levels):
    """
    Computes the passive state's energy of a battery given its density matrix (rho)
    and the energy levels (assumed in ascending order).
    """
    eigvals, _ = rho.eigenstates()
    eigvals = np.array(eigvals, dtype=float)
    sorted_populations = np.sort(eigvals)[::-1]  # descending order
    passive_energy = np.dot(sorted_populations, levels)
    return passive_energy

# Initial battery state
battery_int = basis(N + 1, 0)

# Battery ladder operators
A = sum(basis(N + 1, n - 1) * basis(N + 1, n).dag() for n in range(1, N + 1))  
A_dag = A.dag()

# Interaction Unitaries
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()

ergotropy_data = np.zeros((len(kappa_values), len(p_values)))

for k_idx, kappa in enumerate(kappa_values):
    V_env = np.sqrt(kappa / delta_t) * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U_env = (-1j * V_env).expm()
    
    for p_idx, p_swap in enumerate(p_values):
        prev_env_ancilla = fresh_ancilla_env()
        one_state = ket2dm(tensor(battery_int, basis(2, 1)))
        max_ergotropy = 0
        levels = np.arange(N + 1)
        
        for collision in range(1, collision_num + 1):
            # Battery-Charger interaction
            one_state = U * one_state * U.dag()
            battery = one_state.ptrace(0)
            
            # Partial swap with previous environment ancilla
            combined = tensor(fresh_ancilla_env(), prev_env_ancilla)
            swapped = partial_swap_map(combined, p_swap)
            new_env_ancilla = swapped.ptrace(0)
            
            # Battery-Environment interaction
            one_state_env = tensor(battery, new_env_ancilla)
            one_state_env = U_env * one_state_env * U_env.dag()
            
            # Extract updated battery and env
            battery = one_state_env.ptrace(0)
            updated_env_ancilla = one_state_env.ptrace(1)
            prev_env_ancilla = updated_env_ancilla
            
            # Calculate ergotropy
            probability = battery.diag()
            average_energy = np.dot(levels, probability)
            ergotropy = average_energy - get_passive_energy(battery, levels)
            max_ergotropy = max(max_ergotropy, ergotropy)
            
            # Reset state for next loop
            one_state = tensor(battery, fresh_ancilla())
        
        ergotropy_data[k_idx, p_idx] = max_ergotropy

# Create contour plot
plt.figure(figsize=(10, 6))
X, Y = np.meshgrid( g/kappa_values,p_values)
cp = plt.contourf(X, Y, ergotropy_data, levels=20, cmap='viridis')
plt.colorbar(cp, label='Max Ergotropy')
plt.xlabel("p (Partial Swap Strength)", fontsize=14)
plt.ylabel("g/kappa (Relative Environment Coupling Strength)", fontsize=14)
plt.title("Contour Plot of Max Ergotropy Over Collisions", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

end = time.time()
print("Time Elapsed:", end - start)
print("In minutes:", (end - start) / 60)
