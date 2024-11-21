import numpy as np
import matplotlib.pyplot as plt

# Number of sigma states (|\u03c3_k⟩)
N = 3  # Adjust as needed
# Number of possible outputs (|x⟩)
M = 2  # Adjust as needed

# Define T^x matrices (transition amplitudes)
T_matrices = []
for x in range(M):
    # For each x, define an (N x N) matrix T^x
    T_x = np.random.rand(N, N)  # Replace with specific matrices if needed
    T_matrices.append(T_x)

# Total dimension of the combined Hilbert space
dim = N * M

# Initialize the unitary operator U as a (dim x dim) matrix
U = np.zeros((dim, dim), dtype=complex)

# Construct U based on the given transformation
for k in range(N):
    index_k0 = k * M + 0  # Index for |\u03c3_k⟩|0⟩
    for x in range(M):
        T_x = T_matrices[x]
        for j in range(N):
            T_x_kj = T_x[k, j]
            index_jx = j * M + x  # Index for |\u03c3_j⟩|x⟩
            U[index_jx, index_k0] = T_x_kj

# Function to create basis vectors in the combined Hilbert space
def basis_vector(k, x):
    vec = np.zeros(dim, dtype=complex)
    index = k * M + x
    vec[index] = 1.0
    return vec

# Example: Apply U to |\u03c3_0⟩|0⟩
k = 0  # Choose the sigma state index
psi_k0 = basis_vector(k, 0)  # Initial state |\u03c3_k⟩|0⟩

# Compute the result of U acting on the initial state
result = U @ psi_k0

# Extract probabilities from the resulting state
probabilities = np.abs(result) ** 2

# Plot the resulting state probabilities as a point graph with a line
labels = []
for index in range(dim):
    j = index // M
    x = index % M
    labels.append(f"|\u03c3_{j}\u27e9|{x}\u27e9")

plt.figure(figsize=(10, 6))
plt.plot(labels, probabilities, marker='o', linestyle='-', color='b', alpha=0.7)
plt.xlabel('Basis States')
plt.ylabel('Probability')
plt.title(f'Resulting State Probabilities after Applying U to |\u03c3_{k}\u27e9|0\u27e9')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
