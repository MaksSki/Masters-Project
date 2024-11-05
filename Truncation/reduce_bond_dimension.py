#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:31:58 2024
New updated Schmidt decomposition function for reduction of bond dimensions
@author: maksymilianskiba
"""
def schmidt_decomposition_MPS(MPS, k):
    N = len(MPS)
    
    # Contract left tensors up to site k
    left_tensor = MPS[0]
    for n in range(1, k + 1):
        left_tensor = np.tensordot(left_tensor, MPS[n], axes=([1], [0]))  # Contract shared dimension
        left_tensor = left_tensor.reshape(-1, left_tensor.shape[-1])      # Reshape to a 2D matrix
    
    # Contract right tensors starting from site k+1
    right_tensor = MPS[k + 1]
    for n in range(k + 2, N):
        right_tensor = np.tensordot(right_tensor, MPS[n], axes=([1], [0]))  # Contract shared dimension
        right_tensor = right_tensor.reshape(right_tensor.shape[0], -1)      # Reshape to a 2D matrix
    
    # Combine left and right contracted tensors into a single matrix for SVD
    combined_matrix = np.dot(left_tensor, right_tensor)
    
    # Perform SVD on the combined matrix
    U, S, Vh = np.linalg.svd(combined_matrix, full_matrices=False)
    
    return U, S, Vh

