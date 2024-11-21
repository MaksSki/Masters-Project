# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:20:47 2024

@author: rithi
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import unitary_group
start = time.time()
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0,1],[1,0]])
pauli_z = np.array([[1,0],[0,-1]])
a = np.array([0,0,1])
b1 = np.array([0,1,0])
b2 = np.array([1,0,0])

mapAB = np.array([[0,0,0], [1,0,0], [0,0,1],[0,1,0],[0,0,0],[0,0,0]])

def U_mkl(m,k,l,phi,theta):
    return (np.cos(theta)*np.outer(k,k)) + (np.cos(theta)*np.outer(l,l)) + (np.exp(1j*phi)*np.sin(theta)*np.outer(k,l)) - (np.exp(-1j*phi)*np.sin(theta)*np.outer(l,k)) + np.outer(m,m)

def partial_trace(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])
from mpmath import mp, mpc, matrix, qr
import numpy as np

# Set precision to 100 decimal places
mp.dps = 100

def generate_high_precision_unitary_2x2():
    # Step 1: Generate a random 2x2 complex matrix with high precision
    A = matrix([
        [mpc(np.random.randn(), np.random.randn()), mpc(np.random.randn(), np.random.randn())],
        [mpc(np.random.randn(), np.random.randn()), mpc(np.random.randn(), np.random.randn())]
    ])

    # Step 2: Perform QR decomposition using mpmath to obtain a unitary matrix Q
    Q, R = qr(A)

    # Step 3: Normalize Q to ensure that it is unitary
    for i in range(2):
        if R[i, i] != 0:
            Q[:, i] *= (1 / abs(R[i, i]))

    return Q

def convert_mpmath_to_numpy(mp_matrix):
    """Convert an mpmath matrix to a NumPy array with complex128 elements"""
    rows, cols = mp_matrix.rows, mp_matrix.cols
    np_matrix = np.zeros((rows, cols), dtype=np.complex128)
    
    for i in range(rows):
        for j in range(cols):
            np_matrix[i, j] = complex(mp_matrix[i, j].real, mp_matrix[i, j].imag)
    
    return np_matrix

# Generate the highly accurate unitary 2x2 matrix and convert it
U_mp = generate_high_precision_unitary_2x2()
U = convert_mpmath_to_numpy(U_mp)

# Initial condition
dt = 0.1

# Testing for Random Unitaries - choose one approach
# U = unitary_group.rvs(2)   # Random unitary using scipy
# U = random_unitary(2)      # Random unitary from custom function
# U = np.cos(dt)**2 * pauli_y - np.sin(dt)**2 * pauli_y  # Alternative definition

# Save U for consistency in testing (optional)
np.save("shared_U.npy", U)

sys_state = np.array([1, 0])
a = np.array([0, 0, 1])
state_initial = np.kron(sys_state, a)

# Updated total operator using the converted U
total_operator = np.kron(np.kron(U, np.eye(3)), U)  # U x I(3) x U

# Additional setup
phi_init = 0
theta_init = np.pi / 4
len_states = 10

# Data collection for plotting
data = []
steps = np.arange(1, len_states + 1, step=1)

for i in range(len_states):
    if i == 0:
        # Generation
        U_i = np.kron(np.eye(2), U_mkl(np.array([0, 1, 0]), a, np.array([1, 0, 0]), phi_init, theta_init))
        mapAB_i = np.kron(np.eye(2), mapAB)
        state = np.matmul(mapAB_i, np.matmul(U_i, state_initial))
        state = np.matmul(total_operator, state)

        memory = state

        # Uncompute
        state = state.reshape(6, 2)
        U, S, Vh = np.linalg.svd(state, full_matrices=False)

        Vh_dag = np.conjugate(Vh).T
        state = np.dot(U @ np.diag(S), Vh @ Vh_dag)
        state = state.flatten()

    elif i == len_states - 1:
        U_i = np.kron(np.eye(2), np.kron(U_mkl(a, np.array([0, 1, 0]), np.array([1, 0, 0]), 0, np.pi / 2), np.eye(6)))
        mapAB_i = np.kron(np.eye(2), np.kron(mapAB, np.eye(6)))
        state = np.matmul(mapAB_i, np.matmul(U_i, state))

        total_operator_i = np.kron(total_operator, np.eye(6))
        state = np.matmul(total_operator_i, state)

        memory = state

        # Uncompute
        state = state.reshape(6, 2 * 6)
        U, S, Vh = np.linalg.svd(state, full_matrices=False)

        Vh_dag = np.conjugate(Vh).T
        state = np.dot(np.dot(U, np.diag(S)), Vh @ Vh_dag)
        state = state.flatten()

    # Simplified for other steps...

    if len(state) == 36:
        rho = np.outer(state, state)
        sys = partial_trace(rho, [2, 18], axis=1)
        exp_sig = np.trace(np.matmul(sys, pauli_z))
        data.append(exp_sig.item())
    else:
        rho = np.outer(state, state)
        sys = partial_trace(rho, [2, 3 * 2 ** (i + 1)], axis=1)
        exp_sig = np.trace(np.matmul(sys, pauli_z))
        data.append(exp_sig.item())

# Plotting section remains the same...



# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

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

#plt.savefig("Evolution of expectation pauli z.jpg",dpi=200)
# Show the plot
plt.show()