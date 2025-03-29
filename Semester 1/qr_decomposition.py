
"" "
20 / 10 / 25
@ author:Maksymilian Skiba
Quantum Information Theory Perspective on Structure and Complexity
QR Decomposition in Computational Form
Schmitt Decomposition
"" "

import numpy as np

def state_to_MPS(state_tensor):
N = state_tensor.ndim
MPS =[]
tensor = state_tensor.copy()

for n
in range(N - 1):
#Reshape tensor into matrix
	dim_left = tensor.shape[0]
		dim_right = int	(tensor.size / dim_left)
		tensor_matrix = tensor.
reshape(dim_left, dim_right)

#Perform SVD
U, S, Vh = np.linalg.svd(tensor_matrix, full_matrices = False)

#Form MPS tensor at site n
bond_dim = S.size
mps_tensor = U
#Shape: (dim_left, bond_dim)
MPS.append(mps_tensor)

#Prepare tensor for next iteration
tensor = np.dot(np.diag(S), Vh)
tensor = tensor.reshape(bond_dim, *tensor.shape[1:])
#Correct reshape for next iteration

#Last MPS tensor
MPS.append(tensor)

return MPS

#Parameters
N = 8
d = 10

#Define a general state vector
np.random.seed(42)
state_vector = np.random.rand(d * *N) + 1 j * np.random.rand(d * *N)
state_vector /= np.linalg.norm(state_vector)

#Reshape into state tensor
state_tensor = state_vector.reshape([d] * N)

#Convert to MPS
MPS = state_to_MPS(state_tensor)

#Perform Schmidt Decomposition at partition k
def schmidt_decomposition_MPS(MPS, k):
N = len(MPS)

#Contract left tensors
left_tensor = MPS[0]
for n
in range(1, k + 1):
	left_tensor = np.tensordot(left_tensor, MPS[n], axes = ([1],[0]))
		left_tensor = left_tensor.reshape(-1, left_tensor.shape[-1])
#Merge indices

#Contract right tensors
		right_tensor = MPS[k + 1]
		for n
in range(k + 2, N):
		right_tensor = np.tensordot(right_tensor, MPS[n], axes = ([1],[0]))
			right_tensor = right_tensor.reshape(right_tensor.shape[0], -1)
#Merge indices

#Combine left and right tensors
			combined_matrix = np.dot(left_tensor, right_tensor)

#Perform SVD
			U, S, Vh = np.linalg.svd(combined_matrix, full_matrices = False)

			return U, S, Vh

#Choose partition
			k = 1
#Between qubits 1 and 2

#Perform Schmidt Decomposition
			U, S, Vh = schmidt_decomposition_MPS(MPS, k)

			print("Schmidt coefficients:", S)
			print('Sum of Squares', np.sum(S * *2))

def truncation(schmitt_coeff, threshold):
		iterative_sq_sum = 0
			schmitt_coeff_array_point = 0
			while iterative_sq_sum
	<threshold:
			iterative_sq_sum = +np.square(schmitt_coeff[0])
				schmitt_coeff_array_point = +1

				norm_const = np.reciprocal(iterative_sq_sum)
	unnorm_new_schmitt = (schmitt_coeff[:schmitt_coeff_array_point])
				new_schmitt_coeff = unnorm_new_schmitt * norm_const

				return new_schmitt_coeff, norm_const

	def chain_unpacking(MPS, k, threshold):
			N = len(MPS)
				all_truncated_coeffs =[]


				for k
		in range(N - 1):

				U, S, Vh = schmidt_decomposition_MPS(MPS, k)


					truncated_coeffs, norm_const = truncation(S, threshold)
					all_truncated_coeffs.append(truncated_coeffs)


					return all_truncated_coeffs


					threshold = 0.9
					truncated_coefficients = chain_unpacking(MPS, k, threshold)

					print("All truncated Schmidt coefficients for each partition:")
					for idx
			,coeffs in enumerate(truncated_coefficients):
					print(f "Partition {idx}: {coeffs}")
