import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import time
start=time.time()
# Global Variables
collision_num = 3000   # Total number of collisions
n_batt = 5             # Number of batteries in the chain
N = 200                # Battery Hilbert space dimension (levels 0,...,N)
theta = np.pi / 4      # Interaction parameter controlling energy transfer
q = 0.75               # Probability for the qubit to be in the excited state |1>
c = 1                  # Coherence parameter for the ancilla
g_range = np.linspace(0.1,2,39) # Interaction strength range

# Battery unplug threshold
unplug_threshold = 0.8 * N  

A = sum(basis(N+1, n-1) * basis(N+1, n).dag() for n in range(1, N+1))
A_dag = A.dag()

# Function to generate a fresh ancilla state
def fresh_ancilla():
    """Generates a fresh ancilla state defined as a mixture with coherence."""
    rho_Q = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    coherence_term = c * np.sqrt(q * (1 - q)) * (
        basis(2, 1) * basis(2, 0).dag() + basis(2, 0) * basis(2, 1).dag()
    )
    rho_Q = rho_Q + coherence_term
    return rho_Q / rho_Q.tr()

# Initialize final_coll as a NumPy object array to hold lists of unplug events
final_coll = np.empty((len(g_range), n_batt), dtype=object)
for i in range(len(g_range)):
    for j in range(n_batt):
        final_coll[i, j] = []  # Each cell will store a list of unplug events

# Simulate the battery collisions over different interaction strengths
plt.figure(figsize=(10, 6))
for g_index, g in enumerate(g_range):
    # Define the interaction Hamiltonian and unitary evolution operator for this g
    V = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U = (-1j * theta * V).expm()  # Unitary evolution operator
    
    # Initialize batteries and status flags
    batteries = [ket2dm(basis(N+1, 0)) for _ in range(n_batt)]
    plugged = [True] * n_batt  # Track which batteries are active
    avg_energy = [[] for _ in range(n_batt)]
    collisions = []
    
    for collision in range(1, collision_num + 1):
        collisions.append(collision)
        current_ancilla = fresh_ancilla()  # Start with a fresh ancilla
        
        # Propagate the ancilla through the battery chain
        for i in range(n_batt):
            if plugged[i]:
                combined_state = tensor(batteries[i], current_ancilla)
                evolved_state = U * combined_state * U.dag()
                new_battery_state = evolved_state.ptrace(0)
                avg_E = np.dot(np.arange(N+1), new_battery_state.diag())
                batteries[i] = new_battery_state
                
                # Unplug battery if threshold is reached
                if avg_E >= unplug_threshold:
                    plugged[i] = False
                    print(f"Battery {i+1} unplugged at collision {collision} for g={g}")
                    final_coll[g_index, i].append(collision)
                    current_ancilla = fresh_ancilla()  # Reset ancilla after unplug event
                else:
                    current_ancilla = evolved_state.ptrace(1)  # Pass the ancilla on
            else:
                current_ancilla = fresh_ancilla()  # Use a fresh ancilla if battery is unplugged
        
        # Record the average energy for each battery
        for i in range(n_batt):
            avg_E = np.dot(np.arange(N+1), batteries[i].diag())
            avg_energy[i].append(avg_E)
    
    # Plot average energy vs. collisions for each battery at current g
    for i in range(n_batt):
        plt.plot(collisions, avg_energy[i], label=f"Battery {i+1} (g={g:.2f})", linewidth=2)

plt.xlabel("Number of Collisions", fontsize=14)
plt.ylabel("Average Energy Level", fontsize=14)
plt.title("Average Energy vs. Collisions for a Chain of Batteries", fontsize=16)
plt.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# Initialize arrays to hold the theoretical (Markov) predictions and the Markov ratios.
Markov_array = np.empty((len(g_range), n_batt), dtype=object)
Markov_ratio = np.empty((len(g_range), n_batt), dtype=object)

for g_index in range(len(g_range)):
    # Gather the first unplug event (if any) for each battery in this g row.
    unplug_events = []
    for i in range(n_batt):
        if final_coll[g_index, i]:
            unplug_events.append(final_coll[g_index, i][0])
    
    if unplug_events:
        # Define m as the first (smallest) unplug collision number in the row.
        m = min(unplug_events)
        # Build the theoretical array: [m, 2*m, 3*m, ..., n_batt*m]
        theoretical_row = [(i + 1) * m for i in range(n_batt)]
        for i in range(n_batt):
            if final_coll[g_index, i]:
                observed = final_coll[g_index, i][0]
                Markov_array[g_index, i] = theoretical_row[i]
                Markov_ratio[g_index, i] = theoretical_row[i] / observed
            else:
                # If a battery never unplugged, record NaN.
                Markov_array[g_index, i] = np.nan
                Markov_ratio[g_index, i] = np.nan
    else:
        # If no battery unplugged for this g, fill with NaN.
        for i in range(n_batt):
            Markov_array[g_index, i] = np.nan
            Markov_ratio[g_index, i] = np.nan

# Print the computed arrays for verification.
print("\nMarkov Array:")
for g_index, g in enumerate(g_range):
    print(f"g = {g:.2f}: {Markov_array[g_index]}")

print("\nMarkov Ratio:")
for g_index, g in enumerate(g_range):
    print(f"g = {g:.2f}: {Markov_ratio[g_index]}")


plt.figure(figsize=(8, 6))
for batt in range(n_batt):
    ratio_values = []
    for g_index in range(len(g_range)):
        # Directly use the float value from Markov_ratio
        ratio_val = Markov_ratio[g_index, batt]
        ratio_values.append(ratio_val)
    plt.plot(g_range, ratio_values, marker='o', linewidth=2, label=f"Battery {batt+1}")

plt.xlabel("Interaction Strength (g)", fontsize=14)
plt.ylabel("Markov Ratio (Time of Markov/Non Markov Collisions)", fontsize=14)
plt.title("Non Markovian Differences in Charging Multiple Batteries", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
end=time.time()
print('Code Runtime: ',(end-start))