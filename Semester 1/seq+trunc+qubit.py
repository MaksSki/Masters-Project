#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:08:38 2024

@author: maksymilianskiba
"""

# -*- coding: utf-8 -*-
"""
Simplified Sequential Generation and MPS Truncation Script for a Qubit System
"""

import numpy as np

# Unitary Transformation for a Qubit System and Ancilla
def partial_swap(theta):
    """Generate a partial swap unitary matrix with interaction angle theta for two qubits."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    swap_operator = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]], dtype=complex)
    U = cos_theta * np.eye(4, dtype=complex) + 1j * sin_theta * swap_operator
    return U

# Partial Trace Function for a Qubit System
def partial_trace(rho, dim_A, dim_B):
    """Trace out subsystem B from a combined density matrix of two qubits."""
    # Initialize reduced density matrix for subsystem A
    reduced_rho = np.zeros((dim_A, dim_A), dtype=complex)
    for i in range(dim_B):
        reduced_rho += rho[i*dim_A:(i+1)*dim_A, i*dim_A:(i+1)*dim_A]
    return reduced_rho

# MPS Conversion Function from Truncation_chain.py
def state_to_MPS(state_tensor):
    N = state_tensor.ndim
    MPS = []
    tensor = state_tensor.copy()

    for n in range(N - 1):
        dim_left = tensor.shape[0]
        dim_right = int(tensor.size / dim_left)
        tensor_matrix = tensor.reshape(dim_left, dim_right)
        U, S, Vh = np.linalg.svd(tensor_matrix, full_matrices=False)
        MPS.append(U)
        tensor = np.dot(np.diag(S), Vh)
        tensor = tensor.reshape(S.size, *tensor.shape[1:])
    
    MPS.append(tensor)
    return MPS

# Schmidt Decomposition and Truncation
def schmidt_decomposition_MPS(MPS, k):
    N = len(MPS)
    left_tensor = MPS[0]
    for n in range(1, k + 1):
        left_tensor = np.tensordot(left_tensor, MPS[n], axes=([1], [0]))
        left_tensor = left_tensor.reshape(-1, left_tensor.shape[-1])

    right_tensor = MPS[k + 1]
    for n in range(k + 2, N):
        right_tensor = np.tensordot(right_tensor, MPS[n], axes=([1], [0]))
        right_tensor = right_tensor.reshape(right_tensor.shape[0], -1)
    
    combined_matrix = np.dot(left_tensor, right_tensor)
    U, S, Vh = np.linalg.svd(combined_matrix, full_matrices=False)
    return U, S, Vh

def truncation(schmidt_coeff, threshold):
    total_sq_sum = np.sum(np.abs(schmidt_coeff)**2)
    threshold_sum = threshold * total_sq_sum
    iterative_sq_sum = 0
    cutoff_index = 0
    
    for i, coeff in enumerate(schmidt_coeff):
        iterative_sq_sum += np.abs(coeff)**2
        if iterative_sq_sum >= threshold_sum:
            cutoff_index = i + 1
            break
    
    truncated_coeffs = schmidt_coeff[:cutoff_index]
    normalization_factor = np.sqrt(np.sum(np.abs(truncated_coeffs)**2))
    truncated_coeffs_normalized = truncated_coeffs / normalization_factor
    return truncated_coeffs_normalized, iterative_sq_sum / total_sq_sum

# Chain Unpacking Function
def chain_unpacking(MPS, threshold):
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

# Main Execution
if __name__ == "__main__":
    # Initialize the principal qubit system in a superposition state |+⟩ = (|0⟩ + |1⟩)/√2
    initial_system_state = (1 / np.sqrt(2)) * np.array([[1], [1]], dtype=complex)
    rho_system = initial_system_state @ initial_system_state.conj().T

    # Initialize the ancilla (environment qubit) in the ground state |0⟩
    initial_ancilla_state = np.array([[1], [0]], dtype=complex)
    rho_ancilla = initial_ancilla_state @ initial_ancilla_state.conj().T

    num_collisions = 8
    theta = np.pi / 8  # Interaction strength

    for i in range(num_collisions):
        # Apply the unitary interaction
        U = partial_swap(theta)
        rho_combined = np.kron(rho_system, rho_ancilla)
        rho_combined = U @ rho_combined @ U.conj().T

        # Trace out the ancilla to get the updated system state
        rho_system = partial_trace(rho_combined, dim_A=2, dim_B=2)

        # Convert the reduced density matrix to MPS
        MPS = state_to_MPS(rho_system)

        # Set a truncation threshold and truncate MPS terms
        threshold = 0.9
        truncated_info = chain_unpacking(MPS, threshold)
        
        print(f"Collision {i+1} - MPS Terms and Truncated Coefficients:")
        for info in truncated_info:
            print(f"Partition {info['partition']}:")
            print(f"  MPS term: {info['MPS_term']}")
            print(f"  Truncated Schmidt coefficients: {info['truncated_coeffs']}")
        print("-" * 40)

