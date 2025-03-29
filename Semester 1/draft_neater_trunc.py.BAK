#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"" "
Created on Tue Oct 29 08:39:27 2024

@ author:maksymilianskiba
"" "

import numpy as np
import pandas as pd

def state_to_MPS(state_tensor):
N = state_tensor.ndim
MPS =[]
tensor = state_tensor.copy()

for n
in range(N - 1):
	dim_left = tensor.shape[0]
		dim_right = int	(tensor.size / dim_left)
		tensor_matrix = tensor.
reshape(dim_left, dim_right)

U, S, Vh = np.linalg.svd(tensor_matrix, full_matrices = False)

bond_dim = S.size
mps_tensor = U
#Shape: (dim_left, bond_dim)
MPS.append(mps_tensor)

tensor = np.dot(np.diag(S), Vh)
tensor = tensor.reshape(bond_dim, *tensor.shape[1:])

MPS.append(tensor)

return MPS

def schmidt_decomposition_MPS(MPS, k):
N = len(MPS)

left_tensor = MPS[0]
for n
in range(1, k + 1):
	left_tensor = np.tensordot(left_tensor, MPS[n], axes = ([1],[0]))
		left_tensor = left_tensor.reshape(-1, left_tensor.shape[-1])

		right_tensor = MPS[k + 1]
		for n
in range(k + 2, N):
		right_tensor = np.tensordot(right_tensor, MPS[n], axes = ([1],[0]))
			right_tensor = right_tensor.reshape(right_tensor.shape[0], -1)

			combined_matrix = np.dot(left_tensor, right_tensor)

			U, S, Vh = np.linalg.svd(combined_matrix, full_matrices = False)

			return U, S, Vh

def truncation(schmitt_coeff, threshold):
		cumulative_sq_sum = 0
			retained_indices = 0
			while cumulative_sq_sum
	<threshold and retained_indices < len(schmitt_coeff):
			cumulative_sq_sum += schmitt_coeff[retained_indices] * *2
				retained_indices += 1

				norm_const = np.sqrt(cumulative_sq_sum)
#Normalize by sqrt for proper truncation
	truncated_coeffs = schmitt_coeff[:retained_indices] / norm_const

				return truncated_coeffs, norm_const

	def chain_unpacking(MPS, threshold):
			N = len(MPS)
				all_truncated_info =[]

				for k
		in range(N - 1):
				U, S, Vh = schmidt_decomposition_MPS(MPS, k)

					truncated_coeffs, norm_const = truncation(S, threshold)

					all_truncated_info.append(
								  {
"partition":k,
"MPS_term_shape":MPS[k].shape,
"truncated_coeffs":truncated_coeffs,
"schmidt_rank":len(truncated_coeffs)
})

return all_truncated_info

#Example Parameters
N = 5
d = 3
np.random.seed(42)
state_vector = np.random.rand(d * *N) + 1 j * np.random.rand(d * *N)
state_vector /= np.linalg.norm(state_vector)
state_tensor = state_vector.reshape([d] * N)

#Convert to MPS
MPS = state_to_MPS(state_tensor)

#Set threshold and apply truncation
threshold = 0.9
chain_unpacking(MPS, threshold)
