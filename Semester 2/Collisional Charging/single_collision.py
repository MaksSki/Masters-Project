import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
collision_num = 3  # Number of collisions
g = 1  # Interaction strength
N = 7  # Number of battery levels
dt = 0.1  # Time step
theta=np.pi/4
# Initial state (empty battery and excited ancilla qubit)
battery_int = basis(N+1, 0)  # |0> in battery Hilbert space
ancilla_int = basis(2, 1)  # |e> in qubit Hilbert space
state = tensor(battery_int, ancilla_int)  # Combined system state

# Fix: Convert state to a density matrix
state = state.proj()  # Now state is ρ = |ψ⟩⟨ψ|

# Define the Hamiltonian of the system
A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))  # Lowering operator for battery
V = g * (tensor(A, sigmap()) + tensor(A.dag(), sigmam()))  # Interaction Hamiltonian

# Unitary evolution operator
U = (-1j * V * theta).expm()

# Debugging: Check dimensions
print("State dims:", state.dims)
print("U dims:", U.dims)

# Apply unitary evolution
U_dagg = U.dag()
evolved_state = U * state * U_dagg  # This now works!

print(evolved_state)
