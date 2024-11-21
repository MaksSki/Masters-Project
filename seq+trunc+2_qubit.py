import numpy as np

# Basic unitary interaction: a partial swap
def partial_swap(theta):
    """Generate a simple partial swap unitary matrix for two qubits.
    This function creates a partial swap operation, where `theta` controls
    the "interaction strength" or degree of swapping. A partial swap is 
    useful to simulate weak interactions where two qubits exchange
    information without fully swapping states.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # Define the swap operator for a 2-qubit basis: swaps |01⟩ with |10⟩
    swap_operator = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]], dtype=complex)
    # Construct the unitary operator using the identity and swap terms
    U = cos_theta * np.eye(4, dtype=complex) + 1j * sin_theta * swap_operator
    return U

# Partial trace function to reduce combined state back to the system's state
def partial_trace(rho, dim_A, dim_B):
    """Trace out subsystem B from a combined density matrix of two qubits.
    This operation returns the reduced density matrix for subsystem A,
    effectively discarding the information about subsystem B (ancilla).
    """
    reduced_rho = np.zeros((dim_A, dim_A), dtype=complex)
    # Sum over partial blocks to obtain the reduced density matrix
    for i in range(dim_B):
        reduced_rho += rho[i*dim_A:(i+1)*dim_A, i*dim_A:(i+1)*dim_A]
    return reduced_rho

# MPS conversion function
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

# Truncation function based on Schmidt coefficients
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

# Main Execution
if __name__ == "__main__":
    # Initialize the principal system qubit in |0⟩ state
    initial_system_state = np.array([[1], [0]], dtype=complex)
    rho_system = initial_system_state @ initial_system_state.conj().T

    # Initialize the ancilla (environment qubit) in |1⟩ state
    initial_ancilla_state = np.array([[0], [1]], dtype=complex)
    rho_ancilla = initial_ancilla_state @ initial_ancilla_state.conj().T

    num_collisions = 3  # Reduced for simplicity in proof of concept
    theta = np.pi / 4  # Interaction strength for partial swap

    for i in range(num_collisions):
        # Create the unitary interaction
        U = partial_swap(theta)
        
        # Print the unitary matrix U for this collision
        print(f"Collision {i+1} - Unitary matrix U:\n{U}")

        # Combine the system and ancilla states into a joint density matrix
        rho_combined = np.kron(rho_system, rho_ancilla)
        
        # Apply the unitary interaction
        rho_combined = U @ rho_combined @ U.conj().T
        
        # Trace out the ancilla to get the updated system state
        rho_system = partial_trace(rho_combined, dim_A=2, dim_B=2)

        # Convert to MPS and perform truncation
        MPS = state_to_MPS(rho_system)
        threshold = 0.9
        truncated_info = chain_unpacking(MPS, threshold)
        
        # Print system's state and truncated MPS info after each collision
        print(f"Collision {i+1} - System state:\n{rho_system}")
        print(f"Truncated MPS Terms and Coefficients:")
        for info in truncated_info:
            print(f"  Partition {info['partition']}:")
            print(f"    MPS term: {info['MPS_term']}")
            print(f"    Truncated Schmidt coefficients: {info['truncated_coeffs']}")
        print("-" * 40)
