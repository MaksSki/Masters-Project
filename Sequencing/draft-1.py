#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:41:24 2024

@author: maksymilianskiba
"""

import numpy as np
import matplotlib.pyplot as plt

# Unitary function from your file, using parameters m, k, l, phi, and theta
def U_mkl(m, k, l, phi, theta):
    """Generate a unitary matrix with parameters m, k, l, phi, theta.
    The vectors m, k, and l should be of dimension 2, suitable for 2-qubit interactions.
    """
    return (np.cos(theta) * np.outer(k, k) +
            np.cos(theta) * np.outer(l, l) +
            np.exp(1j * phi) * np.sin(theta) * np.outer(k, l) -
            np.exp(-1j * phi) * np.sin(theta) * np.outer(l, k) +
            np.outer(m, m))

# Partial trace function for reducing combined state to system's state
def partial_trace(rho, dims, axis=0):
    """Takes partial trace over the subsystem defined by 'axis'."""
    dims_ = np.array(dims)
    reshaped_rho = rho.reshape(np.concatenate([dims_, dims_]))
    indices = list(range(reshaped_rho.ndim))
    for i in range(len(dims) - 1, -1, -1):
        if i == axis:
            indices.pop(i)
    traced_out = np.trace(reshaped_rho, axis1=axis, axis2=axis + len(dims))
    return traced_out
def state_to_MPS(state_tensor):
    """Convert a quantum state tensor into a Matrix Product State (MPS) form.
    The MPS decomposition is useful for efficiently representing quantum
    states in systems with a large number of particles. Each tensor in the
    MPS chain corresponds to the local information and entanglement structure.
    """
    N = state_tensor.ndim
    MPS = []
    tensor = state_tensor.copy()

    for n in range(N - 1):
        dim_left = tensor.shape[0]
        dim_right = int(tensor.size / dim_left)
        tensor_matrix = tensor.reshape(dim_left, dim_right)
        U, S, Vh = np.linalg.svd(tensor_matrix, full_matrices=False)
        MPS.append(U)  # U is the left tensor in the MPS form
        tensor = np.dot(np.diag(S), Vh)  # Carry on with remaining part
        tensor = tensor.reshape(S.size, *tensor.shape[1:])
    
    MPS.append(tensor)  # Last tensor as remainder
    return MPS

# Schmidt decomposition function
def schmidt_decomposition_MPS(MPS, k):
    """Perform Schmidt decomposition at the kth partition of the MPS.
    This decomposition quantifies the entanglement across a bipartition
    of the system. U and Vh contain the left and right Schmidt vectors, 
    and S contains the Schmidt coefficients, representing entanglement.
    """
    N = len(MPS)
    left_tensor = MPS[0]
    # Construct left side of Schmidt decomposition
    for n in range(1, k + 1):
        left_tensor = np.tensordot(left_tensor, MPS[n], axes=([1], [0]))
        left_tensor = left_tensor.reshape(-1, left_tensor.shape[-1])

    # Construct right side of Schmidt decomposition
    right_tensor = MPS[k + 1]
    for n in range(k + 2, N):
        right_tensor = np.tensordot(right_tensor, MPS[n], axes=([1], [0]))
        right_tensor = right_tensor.reshape(right_tensor.shape[0], -1)
    
    # Perform SVD on combined matrix to get Schmidt decomposition
    combined_matrix = np.dot(left_tensor, right_tensor)
    U, S, Vh = np.linalg.svd(combined_matrix, full_matrices=False)
    return U, S, Vh
def truncation(schmidt_coeff, threshold):
    """Truncate the Schmidt decomposition based on a given threshold.
    Only the significant Schmidt coefficients are retained based on 
    the threshold, reducing the computational cost while maintaining
    the essential entanglement structure.
    """
    total_sq_sum = np.sum(np.abs(schmidt_coeff)**2)
    threshold_sum = threshold * total_sq_sum
    iterative_sq_sum = 0
    cutoff_index = 0
    
    # Determine number of coefficients needed to meet the threshold
    for i, coeff in enumerate(schmidt_coeff):
        iterative_sq_sum += np.abs(coeff)**2
        if iterative_sq_sum >= threshold_sum:
            cutoff_index = i + 1
            break
    
    # Normalize truncated coefficients
    truncated_coeffs = schmidt_coeff[:cutoff_index]
    normalization_factor = np.sqrt(np.sum(np.abs(truncated_coeffs)**2))
    truncated_coeffs_normalized = truncated_coeffs / normalization_factor
    return truncated_coeffs_normalized, iterative_sq_sum / total_sq_sum

# Chain unpacking to output MPS terms with coefficients
def chain_unpacking(MPS, threshold):
    """Traverse the MPS and compute truncated Schmidt coefficients for each partition.
    This function collects information about the truncated coefficients and structure
    for each bipartition of the MPS chain, allowing us to analyze the entanglement structure.
    """
    N = len(MPS)
    all_truncated_info = []

    for k in range(N - 1):
        U, S, Vh = schmidt_decomposition_MPS(MPS, k)
        truncated_coeffs, norm_const = truncation(S, threshold)
        all_truncated_info.append({
            "partition": k,
            "MPS_term": MPS[k],
            "truncated_coeffs": truncated_coeffs
        })
    return all_truncated_info

# Other helper functions (state_to_MPS, schmidt_decomposition_MPS, truncation, chain_unpacking)
# [Include them here as they were defined in your original code above.]

# Main Execution with plotting
if __name__ == "__main__":
    # Initialize the principal system qubit in |0⟩ state
    initial_system_state = np.array([[1], [0]], dtype=complex)
    rho_system = initial_system_state @ initial_system_state.conj().T

    # Initialize the ancilla (environment qubit) in |1⟩ state
    initial_ancilla_state = np.array([[0], [1]], dtype=complex)
    rho_ancilla = initial_ancilla_state @ initial_ancilla_state.conj().T

    # Define basis vectors for U_mkl as 2-dimensional
    m = np.array([1, 0], dtype=complex)  # Example basis for system qubit
    k = np.array([0, 1], dtype=complex)  # Example basis for ancilla qubit
    l = np.array([1, 0], dtype=complex)  # Alternative basis for ancilla

    phi = np.pi / 4  # Example phase angle
    theta = np.pi / 4  # Example interaction strength

    num_collisions = 3  # Number of system-ancilla interactions (collisions)
    threshold = 0.9  # Threshold for truncation

    # Data collection for plotting
    fidelity_values = []  # To track fidelity changes
    schmidt_values = []   # To track Schmidt coefficients over time

    for i in range(num_collisions):
        # Generate the unitary matrix with the current parameters
        U = U_mkl(m, k, l, phi, theta)
        
        # Print the unitary matrix U for this collision
        print(f"Collision {i+1} - Unitary matrix U:\n{U}")

        # Combine the system and ancilla states into a joint density matrix
        rho_combined = np.kron(rho_system, rho_ancilla)
        
        # Apply the unitary interaction
        rho_combined = U @ rho_combined @ U.conj().T
        
        # Trace out the ancilla to get the updated system state
        rho_system = partial_trace(rho_combined, dims=[2, 2], axis=1)

        # Convert to MPS and perform truncation
        MPS = state_to_MPS(rho_system)
        truncated_info = chain_unpacking(MPS, threshold)
        
        # Compute and store fidelity with respect to the initial state for analysis
        fidelity = np.real(np.trace(initial_system_state @ rho_system))
        fidelity_values.append(fidelity)

        # Collect Schmidt coefficients for plotting
        for info in truncated_info:
            schmidt_values.append(np.real(np.sum(info['truncated_coeffs']**2)))

        # Print system's state and truncated MPS info after each collision
        print(f"Collision {i+1} - System state:\n{rho_system}")
        print(f"Truncated MPS Terms and Coefficients:")
        for info in truncated_info:
            print(f"  Partition {info['partition']}:")
            print(f"    MPS term: {info['MPS_term']}")
            print(f"    Truncated Schmidt coefficients: {info['truncated_coeffs']}")
        print("-" * 40)

    # Plotting the fidelity after each collision
    plt.figure()
    plt.plot(range(1, num_collisions + 1), fidelity_values, marker='o')
    plt.xlabel("Collision Number")
    plt.ylabel("Fidelity with Initial State")
    plt.title("System Fidelity After Each Collision")
    plt.show()

    # Plotting the Schmidt coefficients evolution
    plt.figure()
    plt.plot(range(1, len(schmidt_values) + 1), schmidt_values, marker='o')
    plt.xlabel("Collision Number")
    plt.ylabel("Schmidt Coefficients (Sum of Squares)")
    plt.title("Evolution of Schmidt Coefficients")
    plt.show()
