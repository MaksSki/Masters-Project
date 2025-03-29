import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# We will loop over multiple g values:
g_values = np.linspace(0, 3, 4)  # e.g., 4 points: 0.0, 1.0, 2.0, 3.0

collision_num = 1000      # Total number of collisions
N = 200                   # Number of battery energy levels (levels 0,...,N)
theta = np.pi / 4         # Interaction parameter controlling energy transfer
q = 0.75                  # Probability for qubit to be in ground state (|0>)
c = 1                     # Coherence parameter

# We will store the collision indices just once for plotting
collisions_array = np.arange(1, collision_num + 1)

plt.figure(figsize=(10, 6))

for g in g_values:
    print(f"Running simulation for g = {g}")

    unplug_threshold = 0.85 * N

    # -- Define Initial States for each run --
    battery_int_1 = basis(N + 1, 0)
    battery1_state = ket2dm(battery_int_1)
    battery_2 = ket2dm(basis(N + 1, 0))
    first_ancilla = basis(2, 1)

    # -- Define Operators that depend on g --
    A = sum(basis(N + 1, n - 1) * basis(N + 1, n).dag() for n in range(1, N + 1))
    A_dag = A.dag()
    V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U = (-1j * theta * V).expm()

    # -- Combine battery 1 & ancilla into one_state --
    one_state = tensor(battery_int_1, first_ancilla)

    # -- Tracking arrays for plotting --
    avg_energy_battery1 = []
    avg_energy_battery2 = []

    # Flag to indicate if Battery 1 is unplugged
    battery1_unplugged = False

    # -- Main collision loop --
    for collision in range(1, collision_num + 1):

        U_dagg = U.dag()

        if not battery1_unplugged:
            # Battery 1 interacts with ancilla
            one_state = U * one_state * U_dagg
            battery1_state = one_state.ptrace(0)  # Extract Battery 1 state
            avg1 = np.dot(np.arange(N + 1), battery1_state.diag())

            # Check unplug condition
            if avg1 >= unplug_threshold:
                battery1_unplugged = True
                print(f"Battery 1 unplugged at collision {collision} for g={g}")

            # Prepare fresh ancilla for Battery 1
            rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
            coherence_term = c * np.sqrt(q * (1 - q)) * (
                basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
            )
            rho_Q = rho_Q + coherence_term
            rho_Q = rho_Q / rho_Q.tr()

            # The leftover ancilla from Battery 1
            leftover_ancilla = one_state.ptrace(1)

            # Reset Battery 1 with fresh ancilla
            one_state = tensor(battery1_state, rho_Q)

            # Battery 2 interacts with leftover ancilla
            two_state = tensor(battery_2, leftover_ancilla)
            two_state = U * two_state * U_dagg
            battery_2 = two_state.ptrace(0)
            avg2 = np.dot(np.arange(N + 1), battery_2.diag())

        else:
            # Battery 1 is unplugged; no more collisions for battery 1
            avg1 = np.dot(np.arange(N + 1), battery1_state.diag())

            # But Battery 2 still interacts with a fresh ancilla
            rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
            coherence_term = c * np.sqrt(q * (1 - q)) * (
                basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
            )
            rho_Q = rho_Q + coherence_term
            rho_Q = rho_Q / rho_Q.tr()

            two_state = tensor(battery_2, rho_Q)
            two_state = U * two_state * U_dagg
            battery_2 = two_state.ptrace(0)
            avg2 = np.dot(np.arange(N + 1), battery_2.diag())

        # Record the average energies for this collision
        avg_energy_battery1.append(avg1)
        avg_energy_battery2.append(avg2)

    # -- Plot results for this g value on the same figure --
    plt.plot(
        collisions_array,
        avg_energy_battery1,
        label=f"Battery 1 (g={g:.2f})",
        linewidth=1.5
    )
    plt.plot(
        collisions_array,
        avg_energy_battery2,
        label=f"Battery 2 (g={g:.2f})",
        linewidth=1.5,
        linestyle="--"
    )

# -- Final plot cosmetics --
plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Average Energy Level", fontsize=14)
plt.title("Average Energy vs. Number of Collisions", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
