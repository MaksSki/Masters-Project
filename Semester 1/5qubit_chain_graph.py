#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:14:16 2024

@author: maksymilianskiba
"""

import numpy as np
import matplotlib.pyplot as plt
import time
# Modified __main__ block to ensure compatibility with the intended purpose

def U_mkl(m, k, l, phi, theta):
    return (np.cos(theta) * np.outer(k, k)) + (np.cos(theta) * np.outer(l, l)) + \
           (np.exp(1j * phi) * np.sin(theta) * np.outer(k, l)) - \
           (np.exp(-1j * phi) * np.sin(theta) * np.outer(l, k)) + np.outer(m, m)

def partial_trace(rho, dims, axis=0):
    dims_ = np.array(dims)
    reshaped_rho = rho.reshape(np.concatenate([dims_, dims_]))
    indices = list(range(reshaped_rho.ndim))
    for i in range(len(dims) - 1, -1, -1):
        if i == axis:
            indices.pop(i)
    traced_out = np.trace(reshaped_rho, axis1=axis, axis2=axis + len(dims))
    return traced_out

def state_to_MPS(state_tensor):
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
    total_sq_sum = np.sum(np.abs(schmidt_coeff) ** 2)
    threshold_sum = threshold * total_sq_sum
    iterative_sq_sum = 0
    cutoff_index = 0

    for i, coeff in enumerate(schmidt_coeff):
        iterative_sq_sum += np.abs(coeff) ** 2
        if iterative_sq_sum >= threshold_sum:
            cutoff_index = i + 1
            break

    truncated_coeffs = schmidt_coeff[:cutoff_index]
    normalization_factor = np.sqrt(np.sum(np.abs(truncated_coeffs) ** 2))
    truncated_coeffs_normalized = truncated_coeffs / normalization_factor
    return truncated_coeffs_normalized, iterative_sq_sum / total_sq_sum

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



if __name__ == '__main__':
    # Start counting the time of the program running
    start= time.time()
    # Initialize parameters for the unitary transformation
    phi = np.pi / 4
    theta = np.pi / 6
    dim = 4
    chain_length = 5  # Longer chain with 5 qubits

    # Generate initial state tensor for principal system and ancillae
    # Using a random tensor to create a more diverse entanglement structure.
    state_tensor = np.random.rand(*(dim,) * chain_length)

    # Convert state to MPS form
    MPS = state_to_MPS(state_tensor)

    # Display the components of the MPS
    print("\nComponents of the MPS:")
    for idx, mps_component in enumerate(MPS):
        print(f"MPS Component {idx}:")
        print(mps_component)

    # Iterate over each ancilla with a unitary transformation
    threshold = 0.9  # Truncation threshold
    all_truncated_coeffs = []  # Store Schmidt coefficients for each ancilla

    for ancilla_index in range(1, len(MPS)):
        # Apply unitary transformation on principal system and current ancilla
        k_vector = MPS[ancilla_index - 1][:, 0]
        l_vector = MPS[ancilla_index][:, 0]
        m_vector = np.random.rand(dim)  # Using a random vector for m to diversify the state

        unitary_result = U_mkl(m_vector, k_vector, l_vector, phi, theta)
        
        # Decompose the updated MPS using Schmidt decomposition and truncation
        U, S, Vh = schmidt_decomposition_MPS(MPS, ancilla_index - 1)
        truncated_coeffs, _ = truncation(S, threshold)
        all_truncated_coeffs.append(truncated_coeffs)

        # Print information on the current state
        print(f"\nAncilla {ancilla_index}: Truncated Schmidt coefficients:", truncated_coeffs)

    # Final unpacking of the chain to analyze entanglement
    truncated_info = chain_unpacking(MPS, threshold)
    print("\nFinal truncated information for each partition:")
    for info in truncated_info:
        print(f"Partition {info['partition']}: Truncated Coefficients: {info['truncated_coeffs']}")

    # Plotting Schmidt coefficients for each partition
    plt.figure(figsize=(10, 6))
    for ancilla_index, truncated_coeffs in enumerate(all_truncated_coeffs):
        plt.plot(truncated_coeffs, label=f'Ancilla {ancilla_index + 1}')

    plt.xlabel('Index of Schmidt Coefficient')
    plt.ylabel('Value of Schmidt Coefficient')
    plt.title('Truncated Schmidt Coefficients for Each Ancilla')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    end=time.time()
    print('Time of Program running:',end-start)
