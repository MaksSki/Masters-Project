import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# Parameters
collision_num = 1000    # Number of collisions
record_step = 1         # (Unused but could be used for recording data)
g = 1                   # Battery-charger coupling
theta = np.pi / 4       # Interaction time parameter
kappa = 2               # Environment coupling
delta_t = 1             # Duration of each collision (time-step)
omega0 = 1              # Battery frequency (ω₀)
# Parameter for the ancilla (charger) state coherence
c = 1                 
q = 0.75              
q_env = 0.5          

# Define functions
def fresh_ancilla():
    """Prepare the charger ancilla state (a mixture with coherence)."""
    # Here we use a qubit state as in the paper.
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag())
    rho_Q = rho_Q + coherence_term
    return rho_Q / rho_Q.tr()

def fresh_ancilla_env():
    """Prepare the environment ancilla state (typically ground state)."""
    return ket2dm(basis(2, 0))

def partial_swap_map(rho, p):
    """Partial swap quantum map."""
    swap_op = swap(2,2)  # Swap operator on two qubits
    rho_new = (1 - p) * rho + p * swap_op * rho * swap_op.dag()
    return rho_new / rho_new.tr()

# In this updated version the battery is a qubit.
battery_dim = 2  
# Battery initial state: assume the battery starts in its lower-energy state.
battery_int = basis(battery_dim, 1)  

# The charger is also a qubit; we initialize it using the fresh ancilla function.
charger_init = fresh_ancilla()

# Define the battery-charger interaction Hamiltonian.
# According to the paper, H_BC = g*(σ⁻_B ⊗ σ⁺_C + σ⁺_B ⊗ σ⁻_C)
V = g * (tensor(sigmam(), sigmap()) + tensor(sigmap(), sigmam()))
U = (-1j * theta * V).expm()

# Define the charger-environment interaction Hamiltonian.
# Here we use a similar structure, with the (now qubit) charger replaced by the ancilla that interacts with the environment.
V_env = np.sqrt(kappa / delta_t) * (tensor(sigmap(), sigmam()) + tensor(sigmam(), sigmap()))
U_env = (-1j * V_env).expm()

# Prepare lists to record battery energy and ergotropy.
energy_list = []
ergotropy_list = []
collisions = []

# Loop over different values of the swap probability p (which quantifies memory effects)
plt.figure(figsize=(10, 6))
for p_swap in np.linspace(0, 1, 11):  # p from 0 to 1 in steps of 0.1
    # Initialize joint battery+charger state
    one_state = tensor(battery_int, fresh_ancilla_env())
    new_env_ancilla = fresh_ancilla_env()
    
    # Clear recorded values for each p_swap
    energy_vals = []
    ergotropy_vals = []
    coll = []
    
    for collision in range(1, collision_num + 1):
        # Battery-charger interaction
        one_state = U * one_state * U.dag()
        battery = one_state.ptrace(0)  # Extract battery state
        
        # Charger-environment interaction:
        # Replace the charger state by letting it interact with an environment ancilla.
        one_state_env = tensor(one_state.ptrace(1), new_env_ancilla)
        one_state_env = U_env * one_state_env * U_env.dag()
        # Update charger state from the interaction
        updated_env_ancilla = one_state_env.ptrace(1)
        
        # Implement ancilla-ancilla (environment) collision via a partial swap:
        combined = tensor(fresh_ancilla_env(), updated_env_ancilla)
        swapped = partial_swap_map(combined, p_swap)
        new_env_ancilla = swapped.ptrace(0)
        
        # Recompose the battery+charger state:
        # We keep the battery state unchanged and update the charger with a fresh ancilla.
        
        one_state = tensor(battery, new_env_ancilla)
        
        # Compute battery energy and ergotropy using qubit formulas.
        # Energy: E = (ω₀/2) * ⟨σ_z⟩
        energy = (omega0 / 2) * expect(sigmaz(), battery)
        # Compute the Bloch components
        sx = expect(sigmax(), battery)
        sy = expect(sigmay(), battery)
        sz = expect(sigmaz(), battery)
        norm_sigma = np.sqrt(sx*2 + sy*2 + sz*2)
        # Ergotropy: E_erg = (ω₀/2) * (|⟨σ⟩| + ⟨σ_z⟩)
        ergotropy = (omega0 / 2) * (norm_sigma + sz)
        
        energy_vals.append(energy)
        ergotropy_vals.append(ergotropy)
        coll.append(collision)
        
    plt.plot(coll, ergotropy_vals, label=f"p={p_swap:.1f}", linewidth=2)
    
    # (Optionally, you can also store energy_vals for further comparison.)

plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Ergotropy (in units of ω₀)", fontsize=14)
plt.title("Battery Ergotropy vs. Number of Collisions (for different p)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
