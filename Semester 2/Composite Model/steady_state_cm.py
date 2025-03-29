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
kappa_values = np.linspace(2/3,2,10)
p_values = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])  # Different partial swap strengths

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
steady_state_energy = np.zeros((len(kappa_values), len(p_values)))
steady_state_ergotropy = np.zeros((len(kappa_values), len(p_values)))

for k_idx, kappa in enumerate(kappa_values):
    V_env = np.sqrt(kappa / delta_t) * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U_env = (-1j * V_env).expm()
    
    for p_idx, p_swap in enumerate(p_values):
        prev_env_ancilla = fresh_ancilla_env()
        one_state = ket2dm(tensor(battery_int, basis(2, 1)))
        levels = np.arange(N + 1)
        
        for collision in range(1, collision_num + 1):
            one_state = U * one_state * U.dag()
            battery = one_state.ptrace(0)
            
            combined = tensor(fresh_ancilla_env(), prev_env_ancilla)
            swapped = partial_swap_map(combined, p_swap)
            new_env_ancilla = swapped.ptrace(0)
            
            one_state_env = tensor(battery, new_env_ancilla)
            one_state_env = U_env * one_state_env * U_env.dag()
            
            battery = one_state_env.ptrace(0)
            updated_env_ancilla = one_state_env.ptrace(1)
            prev_env_ancilla = updated_env_ancilla
            
            probability = battery.diag()
            average_energy = np.dot(levels, probability)
            ergotropy = average_energy - get_passive_energy(battery, levels)
            
            if collision == collision_num:
                steady_state_energy[k_idx, p_idx] = average_energy
                steady_state_ergotropy[k_idx, p_idx] = ergotropy

# Ensure correct meshgrid dimensions for plotting
X, Y = np.meshgrid(p_values, g/kappa_values)

# Debugging: Check shape compatibility before plotting
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("steady_state_energy shape:", steady_state_energy.shape)
print("steady_state_ergotropy shape:", steady_state_ergotropy.shape)

# Transpose data if needed
if steady_state_energy.shape != X.shape:
    steady_state_energy = steady_state_energy.T
if steady_state_ergotropy.shape != X.shape:
    steady_state_ergotropy = steady_state_ergotropy.T

# Create contour plot for steady-state energy
plt.figure(figsize=(10, 6))
cp = plt.contourf(X, Y, steady_state_energy, levels=20, cmap='viridis')
plt.colorbar(cp, label='Steady-State Energy')
plt.xlabel("p (Partial Swap Strength)", fontsize=14)
plt.ylabel("kappa (Relative Environment Coupling Strength)", fontsize=14)
plt.title("Contour Plot of Steady-State Energy", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Create contour plot for steady-state ergotropy
plt.figure(figsize=(10, 6))
cp = plt.contourf(X, Y, steady_state_ergotropy, levels=20, cmap='viridis')
plt.colorbar(cp, label='Steady-State Ergotropy')
plt.xlabel("p (Partial Swap Strength)", fontsize=14)
plt.ylabel("kappa (Relative Environment Coupling Strength)", fontsize=14)
plt.title("Contour Plot of Steady-State Ergotropy", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

end = time.time()
print("Time Elapsed:", end - start)
print("In minutes:", (end - start) / 60)
