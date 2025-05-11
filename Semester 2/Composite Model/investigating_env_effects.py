#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from qutip import (
    basis, ket2dm, tensor, Qobj, destroy, create, sigmam, sigmap,
    swap, expect
)

###############################################################################
# 1) SYSTEM PARAMETERS
###############################################################################

# Battery dimension (N+1 levels) -- e.g. a 3-level battery has N=2
N = 3 - 1   # So we have levels 0, 1, 2 for the battery
g = 1.0     # coupling constant for battery–charger
kappa = 2.0 # coupling for battery–environment
delta_t = 1.0
theta = np.pi / 4  # rotation angle for battery–charger partial swap
num_collisions = 30

# We'll sweep partial swap probabilities p for environment memory
p_list = [0.0, 0.2, 0.5, 0.8, 1.0]

###############################################################################
# 2) HELPER FUNCTIONS
###############################################################################

def H_battery_op(N):
    """
    Returns the Hamiltonian for an (N+1)-level battery
    whose nth level has energy n (i.e. equally spaced).
    """
    Hdiag = np.array([float(n) for n in range(N+1)])
    return Qobj(np.diag(Hdiag), dims=[[N+1],[N+1]])

def fresh_ancilla_env():
    """
    Returns the ground-state density matrix of a qubit
    to act as a fresh environment ancilla (|0>).
    """
    return ket2dm(basis(2, 0))

def fresh_ancilla_charger():
    """
    Returns the excited-state density matrix of a qubit (|1>),
    representing the 'charger' ancilla initially in the excited state.
    """
    return ket2dm(basis(2, 1))

def partial_swap_map(rho, p):
    """
    Implements a partial-swap channel on a 2-qubit state rho:
      E(rho) = (1-p)*rho + p * (SWAP * rho * SWAP).
    """
    swap_op = swap(2,2)  # SWAP operator for two qubits
    return (1.0 - p)*rho + p*(swap_op * rho * swap_op)

###############################################################################
# 3) BUILD OPERATORS FOR UNITARIES
###############################################################################

# --- Battery Operators (dimension N+1) ---
a  = destroy(N+1)  # lowering operator
ad = create(N+1)   # raising operator

# --- Charger / Environment Qubit Operators ---
# We'll use QuTiP's sigmap() and sigmam() for raising/lowering.
sm = sigmam()  # sigma-
sp = sigmap()  # sigma+

# H_battery: diagonal with energies 0, 1, 2, ...
H_battery = H_battery_op(N)

# Build the battery–charger interaction Hamiltonian V:
#   V = g * (a ⊗ sigma+ + a† ⊗ sigma-)
V_bc = g * (tensor(a, sp) + tensor(ad, sm))

# Battery–charger collision unitary U_bc = exp(-i * theta * V_bc)
U_bc = ((-1j * theta) * V_bc).expm()

# Build the battery–environment interaction Hamiltonian V_env:
#   V_env = sqrt(kappa / delta_t) * (a ⊗ sigma+ + a† ⊗ sigma-)
g_env = np.sqrt(kappa / delta_t)
V_be = g_env * (tensor(a, sp) + tensor(ad, sm))

# Battery–environment collision unitary U_env = exp(-i * V_be)
U_env = ((-1j) * V_be).expm()

###############################################################################
# 4) COLLISIONAL DYNAMICS FUNCTION
###############################################################################

def run_collisional_model(num_collisions, p):
    """
    Runs battery–charger–environment collisions for a specified partial-swap 'p'.
    Returns a list of battery energies after each collision.
    """

    # (1) INITIAL STATES
    # Battery in ground state
    battery_init = basis(N+1, 0)
    battery_dm   = ket2dm(battery_init)

    # Charger in excited state
    charger_dm = fresh_ancilla_charger()

    # Combine battery & charger into a single density matrix
    one_state = tensor(battery_dm, charger_dm)

    # List to track the average battery energy after each collision
    battery_energies = []

    # Keep track of the "old" environment ancilla (for partial swap memory)
    old_env_ancilla = None

    for n in range(num_collisions):
        # --- Step A: Battery–Charger collision ---
        one_state = U_bc * one_state * U_bc.dag()

        # Extract battery state from battery–charger system
        battery_dm = one_state.ptrace(0)

        # --- Step B: Battery–Environment collision ---
        new_env_ancilla = fresh_ancilla_env()   # environment starts in |0>

        # Combine battery and environment ancilla
        be_state = tensor(battery_dm, new_env_ancilla)

        # Evolve under battery–environment unitary
        be_state = U_env * be_state * U_env.dag()

        # Extract updated battery and environment states
        battery_dm = be_state.ptrace(0)
        updated_env_ancilla = be_state.ptrace(1)

        # --- Step C: Partial Swap with old environment memory ---
        if old_env_ancilla is None:
            # For the first collision, no memory exists; initialize it.
            old_env_ancilla = updated_env_ancilla
        else:
            # Combine the updated environment ancilla with the previous one
            two_qubits = tensor(updated_env_ancilla, old_env_ancilla)
            swapped = partial_swap_map(two_qubits, p)
            # Define:
            # - new_env_ancilla (first qubit) for next collision,
            # - old_env_ancilla (second qubit) to persist as memory.
            new_env_ancilla = swapped.ptrace(0)
            old_env_ancilla = swapped.ptrace(1)
            # (Note: Here new_env_ancilla is not directly used since each collision
            #  starts with a fresh ancilla; only old_env_ancilla carries memory.)

        # --- Step D: Record Battery Energy ---
        energy_val = expect(H_battery, battery_dm)
        battery_energies.append(energy_val)

        # --- Step E: Prepare for the Next Collision ---
        # Re-couple battery with a new charger (in excited state)
        charger_dm = fresh_ancilla_charger()
        one_state = tensor(battery_dm, charger_dm)

    return battery_energies

###############################################################################
# 5) MAIN SCRIPT: SWEEP p, RUN, AND PLOT
###############################################################################

def main():
    results = {}

    for p in p_list:
        energies = run_collisional_model(num_collisions, p)
        results[p] = energies

    # Plot the battery energy versus collision number for each p
    plt.figure(figsize=(7, 5))
    for p in p_list:
        plt.plot(results[p], label=f'p = {p}')
    plt.xlabel('Collision Number')
    plt.ylabel('Battery Energy')
    plt.title('Battery Energy vs. Collisions (Varying Environment Memory p)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
