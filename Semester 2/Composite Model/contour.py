import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ── Global parameters ───────────────────────────────────────────────────────────
collision_num = 5000
n_batt        = 3
N             = 200
theta         = np.pi / 4
q_range       = np.linspace(0, 1, 11)
c             = 1
g_range       = np.linspace(0.1, 2, 20)
unplug_threshold = 0.85 * N

# Ladder operator for the battery (lowering operator)
A      = destroy(N + 1)      # ← fixed line
A_dag  = A.dag()

# ── Helper: fresh ancilla state ─────────────────────────────────────────────────
def fresh_ancilla(q):
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    return rho_Q + coherence

# ── Simulation ─────────────────────────────────────────────────────────────────
final_coll_matrix = np.zeros((len(g_range), len(q_range)))

for i_g, g in enumerate(g_range):
    V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U = (-1j * theta * V).expm()

    for i_q, q in enumerate(q_range):
        batteries = [ket2dm(basis(N + 1, 0)) for _ in range(n_batt)]
        plugged   = [True] * n_batt
        final_collision = collision_num

        for collision in range(1, collision_num + 1):
            ancilla = fresh_ancilla(q)

            for i in range(n_batt):
                if not plugged[i]:
                    ancilla = fresh_ancilla(q)
                    continue

                rho_tot   = tensor(batteries[i], ancilla)
                evolved   = U * rho_tot * U.dag()
                batteries[i] = evolved.ptrace(0)
                avg_E     = np.dot(np.arange(N + 1), batteries[i].diag().real)

                if avg_E >= unplug_threshold:
                    plugged[i] = False
                    if i == n_batt - 1:        # last battery
                        final_collision = collision
                        break
                    ancilla = fresh_ancilla(q)  # reset for next battery
                else:
                    ancilla = evolved.ptrace(1) # pass along

            if not plugged[-1]:
                break

        final_coll_matrix[i_g, i_q] = final_collision

# ── Plot ────────────────────────────────────────────────────────────────────────
G, Q = np.meshgrid(g_range, q_range, indexing='ij')
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(8, 9))
cp = plt.contourf(G, Q, final_coll_matrix, levels=25, cmap='viridis')
plt.colorbar(cp, label='Collision Number for Last Battery Unplug')
plt.xlabel('Interaction Strength (g)', fontsize=18)
plt.ylabel('Qubit Excitation Probability', fontsize=18)
plt.title('Non-Markovian Collisions to a Full Charge for 3 Batteries', fontsize=18)
plt.tight_layout()
plt.show()
