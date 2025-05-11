import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import time

# ──────────────────────────────────────────────────────────
# GLOBAL STYLING
# ──────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "Times New Roman"
# We'll use plt.tick_params after each figure to guarantee the tick labels are sized
# ──────────────────────────────────────────────────────────

start = time.time()

# Global Variables
collision_num = 1000   # Total number of collisions
n_batt        = 3      # Number of batteries in the chain
N             = 200    # Battery Hilbert space dimension
theta         = np.pi/4  # Interaction parameter
q             = 0.75     # Probability for qubit to be in |1>
c             = 1.0      # Coherence parameter
g_range       = np.linspace(0.1, 2, 25)
unplug_thresh = 0.8 * N

# Ladder operators for the battery
A     = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()

def fresh_ancilla():
    """Mixed qubit with coherence."""
    rho_Q = (1-q)*ket2dm(basis(2,1)) + q*ket2dm(basis(2,0))
    coh   = c * np.sqrt(q*(1-q)) * (
        basis(2,1)*basis(2,0).dag() +
        basis(2,0)*basis(2,1).dag()
    )
    rho = rho_Q + coh
    return (rho / rho.tr())

# To store first‐unplug times
final_coll = np.empty((len(g_range), n_batt), dtype=object)
for i in range(len(g_range)):
    for j in range(n_batt):
        final_coll[i,j] = []

# To store average energy histories
avg_energy_all = np.zeros((len(g_range), n_batt, collision_num))

# ──────────────────────────────────────────────────────────
# RUN SIMULATION
# ──────────────────────────────────────────────────────────
for ig, g in enumerate(g_range):
    V = g*(tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U = (-1j * theta * V).expm()

    # initialize batteries
    batteries = [ket2dm(basis(N+1,0)) for _ in range(n_batt)]
    plugged   = [True]*n_batt

    for coll in range(1, collision_num+1):
        anc = fresh_ancilla()
        for b in range(n_batt):
            if plugged[b]:
                joint = tensor(batteries[b], anc)
                evo   = U * joint * U.dag()
                newB  = evo.ptrace(0)
                E     = np.dot(np.arange(N+1), newB.diag())
                batteries[b] = newB

                if E >= unplug_thresh:
                    plugged[b] = False
                    final_coll[ig,b].append(coll)
                    # reset ancilla after unplug
                    anc = fresh_ancilla()
                else:
                    anc = evo.ptrace(1)
            else:
                anc = fresh_ancilla()

            # record energy
            avg_energy_all[ig, b, coll-1] = np.dot(np.arange(N+1), batteries[b].diag())

    print(f"Done g={g:.2f}")

# ──────────────────────────────────────────────────────────
# COMPUTE MARKOV RATIOS
# ──────────────────────────────────────────────────────────
markov_ratio = np.full((len(g_range), n_batt), np.nan)

for ig in range(len(g_range)):
    # find first unplug among all batteries
    unplug_times = [final_coll[ig,b][0] for b in range(n_batt) if final_coll[ig,b]]
    if unplug_times:
        m = min(unplug_times)
        for b in range(n_batt):
            if final_coll[ig,b]:
                obs = final_coll[ig,b][0]
                markov_ratio[ig,b] = ((b+1)*m) / obs

# ──────────────────────────────────────────────────────────
# PLOT 1: Average Energy vs Collisions (all g)
# ──────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.tick_params(axis='both', which='major', labelsize=18)

colls = np.arange(1, collision_num+1)
for ig, g in enumerate(g_range):
    for b in range(n_batt):
        plt.plot(colls,
                 avg_energy_all[ig,b],
                 label=f"Bat {b+1}, g={g:.2f}",
                 linewidth=1.5)

plt.xlabel("Number of Collisions", fontsize=18)
plt.ylabel("Average Energy Level", fontsize=18)
plt.title("Average Energy vs. Collisions for a Chain of Batteries", fontsize=20)
plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────
# PLOT 2: Markov Ratio vs Interaction Strength
# ──────────────────────────────────────────────────────────
plt.figure(figsize=(10, 9))
plt.tick_params(axis='both', which='major', labelsize=18)

for b in range(n_batt):
    plt.plot(g_range,
             markov_ratio[:,b],
             marker='o',
             linewidth=2,
             label=f"Battery {b+1}")

plt.xlabel("Interaction Strength (g)", fontsize=18)
plt.ylabel("Markov Ratio", fontsize=18)
plt.title("Non-Markovian Differences in Charging Multiple Batteries", fontsize=18)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

end = time.time()
print(f"Code Runtime: {end - start:.1f}s")
