#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:06:41 2024

@author: maksymilianskiba
"""

# -*- coding: utf-8 -*-
"""
Combined Sequential Generation and MPS Truncation Script
"""

import numpy as np

# Unitary Transformation Function from seq_gen_ver2.py
def U_mkl(m, k, l, phi, theta):
    return (np.cos(theta) * np.outer(k, k)) + \
           (np.cos(theta) * np.outer(l, l)) + \
           (np.exp(1j * phi) * np.sin(theta) * np.outer(k, l)) - \
           (np.exp(-1j * phi) * np.sin(theta) * np.outer(l, k)) + \
           np.outer(m, m)

# Partial Trace Function from seq_gen_ver2.py
def partial_trace(rho, dims, axis=0):
    dims_ = np.array(dims)
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims) + axis - 1, -1)
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])

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
    a = np.array([0, 0, 1])
    b1 = np.array([0, 1, 0])
    b2 = np.array([1, 0, 0])
    mapAB = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
    
    state_initial = a
    phi_init = 0
    theta_init = np.pi / 4
    len_states = 8

    for i in range(len_states):
        if i == 0:
            state = np.matmul(mapAB, np.matmul(U_mkl(b1, a, b2, phi_init, theta_init), state_initial))
        elif i == len_states - 1:
            U_i = np.kron(np.matmul(U_mkl(a, b1, b2, 0, np.pi / 2), U_mkl(b2, a, b1, 0, np.pi / 2)), np.eye(2**i))
            mapAB_i = np.kron(mapAB, np.eye(2**i))
            state = np.matmul(mapAB_i, np.matmul(U_i, state))
        else:
            U_i = np.kron(U_mkl(b2, a, b1, 0, np.pi / 2), np.eye(2**i))
            mapAB_i = np.kron(mapAB, np.eye(2**i))
            state = np.matmul(mapAB_i, np.matmul(U_i, state))

        dens_op = np.outer(state, state)
        traced_out_rho = partial_trace(dens_op, [3, 2**(i + 1)])

        # Convert to MPS
        MPS = state_to_MPS(traced_out_rho)

        # Set a truncation threshold and truncate MPS terms
        threshold = 0.9
        truncated_info = chain_unpacking(MPS, threshold)
        
        print(f"State {i} MPS Terms and Truncated Coefficients:")
        for info in truncated_info:
            print(f"Partition {info['partition']}:")
            print(f"  MPS term: {info['MPS_term']}")
            print(f"  Truncated Schmidt coefficients: {info['truncated_coeffs']}")
        print("-" * 40)
