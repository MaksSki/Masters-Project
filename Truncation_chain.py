import numpy as np

def state_to_MPS(state_tensor):
    N = state_tensor.ndim
    MPS = []
    tensor = state_tensor.copy()
    
    for n in range(N - 1):
        # Reshape tensor into matrix
        dim_left = tensor.shape[0]
        dim_right = int(tensor.size / dim_left)
        tensor_matrix = tensor.reshape(dim_left, dim_right)
        
        # Perform SVD
        U, S, Vh = np.linalg.svd(tensor_matrix, full_matrices=False)
        
        # Form MPS tensor at site n
        bond_dim = S.size
        mps_tensor = U  # Shape: (dim_left, bond_dim)
        MPS.append(mps_tensor)
        
        # Prepare tensor for next iteration
        tensor = np.dot(np.diag(S), Vh)
        tensor = tensor.reshape(bond_dim, *tensor.shape[1:])  # Correct reshape for next iteration
    
    # Last MPS tensor
    MPS.append(tensor)
    
    return MPS

# Perform Schmidt Decomposition at partition k
def schmidt_decomposition_MPS(MPS, k):
    N = len(MPS)
    
    # Contract left tensors
    left_tensor = MPS[0]
    for n in range(1, k + 1):
        left_tensor = np.tensordot(left_tensor, MPS[n], axes=([1], [0]))
        left_tensor = left_tensor.reshape(-1, left_tensor.shape[-1])  # Merge indices
    
    # Contract right tensors
    right_tensor = MPS[k + 1]
    for n in range(k + 2, N):
        right_tensor = np.tensordot(right_tensor, MPS[n], axes=([1], [0]))
        right_tensor = right_tensor.reshape(right_tensor.shape[0], -1)  # Merge indices
    
    # Combine left and right tensors
    combined_matrix = np.dot(left_tensor, right_tensor)
    
    # Perform SVD
    U, S, Vh = np.linalg.svd(combined_matrix, full_matrices=False)
    
    return U, S, Vh

# Truncation function
def truncation(schmitt_coeff, threshold):
    iterative_sq_sum = 0
    schmitt_coeff_array_point = 0
    while iterative_sq_sum < threshold and schmitt_coeff_array_point < len(schmitt_coeff):
        iterative_sq_sum += schmitt_coeff[schmitt_coeff_array_point] ** 2
        schmitt_coeff_array_point += 1
    
    norm_const = np.reciprocal(iterative_sq_sum)
    unnorm_new_schmitt = schmitt_coeff[:schmitt_coeff_array_point]
    new_schmitt_coeff = unnorm_new_schmitt * norm_const
    
    return new_schmitt_coeff, norm_const

# Chain unpacking to output MPS terms with coefficients
def chain_unpacking(MPS, threshold):
    N = len(MPS)
    all_truncated_info = []
    
    for k in range(N - 1):
        U, S, Vh = schmidt_decomposition_MPS(MPS, k)
        
        truncated_coeffs, norm_const = truncation(S, threshold)
        # Store each MPS term with its associated truncated Schmidt coefficients
        all_truncated_info.append({
            "partition": k,
            "MPS_term": MPS[k],
            "truncated_coeffs": truncated_coeffs
        })
     
    return all_truncated_info

# Parameters
N = 10
d = 6


# Define a general state vector
np.random.seed(42)
state_vector = np.random.rand(d**N) + 1j * np.random.rand(d**N)
state_vector /= np.linalg.norm(state_vector)

# Reshape into state tensor
state_tensor = state_vector.reshape([d]*N)

# Convert to MPS
MPS = state_to_MPS(state_tensor)

# Set the threshold for truncation
threshold = 0.9
truncated_info = chain_unpacking(MPS, threshold)

# Output MPS terms and truncated coefficients for each partition
print("All truncated Schmidt coefficients and MPS terms for each partition:")
for info in truncated_info:
    print(f"Partition {info['partition']}:")
    print(f"  MPS term: {info['MPS_term']}")
    print(f"  Truncated Schmidt coefficients: {info['truncated_coeffs']}")
