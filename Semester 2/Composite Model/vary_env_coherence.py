import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import time

# Parameters
collision_num = 2000
g = 1
N = 200
theta = np.pi / 4
q = 0.75
delta_t = 1
c = 1
temperature = 0.05
start = time.time()

# List of coherence_strength values
coherence_strength_values = [0, 0.1, 0.2, 0.3, 0.4]

# Define a list of p values for the partial swap strength
p_values = np.linspace(0, 1, 20)

def fresh_ancilla():
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q += coherence_term
    return rho_Q.unit()

def realistic_env_state(omega=1.0, coherence_strength=0.1, temperature=temperature):
    k_B = 1.0
    beta = 1 / (k_B * temperature)
    Z = np.exp(-beta * omega / 2) + np.exp(beta * omega / 2)
    
    p0 = np.exp(-beta * omega / 2) / Z  # population of |0⟩
    p1 = np.exp(beta * omega / 2) / Z   # population of |1⟩
    
    max_coh = np.sqrt(p0 * p1)
    coh = min(coherence_strength, max_coh)
    
    rho = Qobj([[p0, coh], [coh, p1]])
    return rho

def partial_swap_map(rho, p):
    swap_op = swap(2, 2)
    rho_new = (1 - p) * rho + p * swap_op * rho * swap_op.dag()
    return rho_new.unit()

def get_passive_energy(rho, levels):
    eigvals, _ = rho.eigenstates()
    eigvals = np.array(eigvals, dtype=float)
    sorted_populations = np.sort(eigvals)[::-1]
    passive_energy = np.dot(sorted_populations, levels)
    return passive_energy

# Initial battery state
battery_int = basis(N+1, 0)

# Battery ladder operators
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()

# Interaction Unitary for battery dynamics (kept fixed)
V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
U = (-1j * theta * V).expm()

levels = np.arange(N+1)

# Prepare plot for peak ergotropy vs. p values
plt.figure(figsize=(10, 6))

# Loop over different coherence strengths
for coh_strength in coherence_strength_values:
    peak_ergotropy_list = []  # To store peak ergotropy for each p value
    
    # Loop over different partial swap strengths (p values)
    for p_swap in p_values:
        # Initialize the environment ancilla with the given coherence strength
        prev_env_ancilla = realistic_env_state(coherence_strength=coh_strength)
        one_state = ket2dm(tensor(battery_int, basis(2, 1)))
        ergo_val = []  # To record ergotropy at each collision
        
        # Run the simulation over many collisions
        for collision in range(1, collision_num + 1):
            # Battery interaction
            one_state = U * one_state * U.dag()
            battery = one_state.ptrace(0)
            
            # Environment collision using the partial swap map with current p_swap
            combined = tensor(realistic_env_state(coherence_strength=coh_strength), prev_env_ancilla)
            swapped = partial_swap_map(combined, p_swap)
            new_env_ancilla = swapped.ptrace(0)
            
            one_state_env = tensor(battery, new_env_ancilla)
            
            # Environment interaction unitary (kept fixed)
            V_env = np.sqrt(1 / delta_t) * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
            U_env = (-1j * V_env).expm()
            one_state_env = U_env * one_state_env * U_env.dag()
            
            battery = one_state_env.ptrace(0)
            updated_env_ancilla = one_state_env.ptrace(1)
            prev_env_ancilla = updated_env_ancilla
            
            # Compute ergotropy
            probability = battery.diag()
            average_energy = np.dot(levels, probability)
            ergotropy = average_energy - get_passive_energy(battery, levels)
            ergo_val.append(ergotropy)
            
            # Update the system state with a fresh ancilla
            one_state = tensor(battery, fresh_ancilla())
        
        # Record the peak ergotropy reached for this p_swap value
        peak_ergotropy_list.append(max(ergo_val))
    
    # Plot peak ergotropy vs. p for the current coherence strength
    plt.plot(p_values, peak_ergotropy_list, label=f"coh_strength={coh_strength:.2f}", linewidth=2)

plt.xlabel("Partial Swap Strength p", fontsize=14)
plt.ylabel("Peak Ergotropy", fontsize=14)
plt.title(f"Peak Ergotropy vs. Partial Swap Strength for Different Coherence Strengths (T={temperature})", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

end = time.time()
print("Time Elapsed:", end - start)
print("In minutes:", (end - start) / 60)
