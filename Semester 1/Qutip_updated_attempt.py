# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:20:47 2024

@author: rithi
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import unitary_group
from qutip import Qobj, ptrace

start = time.time()

a = np.array([0, 0, 1])
b1 = np.array([0, 1, 0])
b2 = np.array([1, 0, 0])

mapAB = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0]])

def U_mkl(m, k, l, phi, theta):
    return (
        (np.cos(theta) * np.outer(k, k))
        + (np.cos(theta) * np.outer(l, l))
        + (np.exp(1j * phi) * np.sin(theta) * np.outer(k, l))
        - (np.exp(-1j * phi) * np.sin(theta) * np.outer(l, k))
        + np.outer(m, m)
    )

def random_unitary(n):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    # Step 2: Perform QR decomposition
    Q, R = np.linalg.qr(A)

    # Step 3: Ensure Q is unitary by adjusting phase
    D = np.diag(R) / np.abs(np.diag(R))
    Q = Q @ np.diag(D.conjugate())

    return Q

# Assuming 'state' is your state vector
# Placeholder code below (replace with your actual state initialization)
state = np.random.rand(6) + 1j * np.random.rand(6)  # Example state vector
state_qobj = Qobj(state)  # Convert state vector to Qobj

rho_qobj = state_qobj * state_qobj.dag()  # Create the density matrix as a Qobj

# Example partial trace operation
if len(state) == 36:
    sys = ptrace(rho_qobj, [1])  # Take partial trace over subsystem 1
else:
    sys = ptrace(rho_qobj, [1])  # Adjust based on dimensions as needed

# Plotting part remains unchanged
# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Placeholder data for plotting (replace with actual data)
steps = np.arange(10)
data = np.random.rand(10)

# Plot the data
ax.plot(steps, data, label="Expectation pauli z", color="blue", linewidth=2)

# Add title and labels with proper font size
ax.set_title("Evolution of system : Hermit Conjugate", fontsize=16, pad=15)
ax.set_xlabel("Steps (collision with ith env state)", fontsize=14)
ax.set_ylabel("Expectation value", fontsize=14)

# Add a grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend with a frame
ax.legend(loc='upper right', fontsize=12, frameon=True)

# Increase the ticks font size for better visibility
ax.tick_params(axis='both', which='major', labelsize=12)

# Set tighter layout for padding
plt.tight_layout()

# plt.savefig("Evolution of expectation pauli z.jpg", dpi=200)
# Show the plot
plt.show()
