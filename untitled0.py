# Adjust run_no_hermit() and run_hermit() functions to ensure dimensional consistency in state evolution

# Modified function for no Hermit Conjugate
def run_no_hermit():
    data = []
    total_operator = np.kron(np.kron(U, np.eye(3)), U)
    state = state_initial
    for i in range(len_states):
        U_i, mapAB_i = None, None  # Reset U_i and mapAB_i for each step

        if i == 0:
            U_i = np.kron(np.eye(2), U_mkl(b1, a, b2, phi_init, theta_init))
            mapAB_i = np.kron(np.eye(2), mapAB)
        elif i == len_states - 1:
            U_i = np.kron(np.eye(2), np.kron(U_mkl(a, b1, b2, 0, np.pi / 2), U_mkl(b2, a, b1, 0, np.pi / 2)))
            mapAB_i = np.kron(np.eye(2), np.kron(mapAB, np.eye(2 ** i)))
        else:
            U_i = np.kron(np.eye(2), np.kron(U_mkl(b2, a, b1, 0, np.pi / 2), np.eye(2 ** i)))
            mapAB_i = np.kron(np.eye(2), np.kron(mapAB, np.eye(2 ** i)))

        # Apply U and mapAB transformations to the state, ensuring proper dimensional consistency
        state = np.matmul(mapAB_i, np.matmul(U_i, state))
        total_operator_i = np.kron(total_operator, np.eye(2 ** i))
        state = np.matmul(total_operator_i, state)

        # Calculate expectation value
        dens_op = np.outer(state, state)
        sys = partial_trace(dens_op, [2, 3 * 2 ** (i + 1)], axis=1)
        exp_sig = np.trace(np.matmul(sys, pauli_z))
        data.append(exp_sig.item())

    return data

# Modified function for Hermit Conjugate
def run_hermit():
    data = []
    total_operator = np.kron(np.kron(U, np.eye(3)), U)
    state = state_initial
    for i in range(len_states):
        U_i, mapAB_i = None, None

        if i == 0:
            U_i = np.kron(np.eye(2), U_mkl(b1, a, b2, phi_init, theta_init))
            mapAB_i = np.kron(np.eye(2), mapAB)
        elif i == len_states - 1:
            U_i = np.kron(np.eye(2), np.kron(U_mkl(a, b1, b2, 0, np.pi / 2), U_mkl(b2, a, b1, 0, np.pi / 2)))
            mapAB_i = np.kron(np.eye(2), np.kron(mapAB, np.eye(6)))
        else:
            U_i = np.kron(np.eye(2), np.kron(U_mkl(b2, a, b1, 0, np.pi / 2), np.eye(6)))
            mapAB_i = np.kron(np.eye(2), np.kron(mapAB, np.eye(6)))

        # Apply U and mapAB transformations to the state
        state = np.matmul(mapAB_i, np.matmul(U_i, state))
        total_operator_i = np.kron(total_operator, np.eye(6))
        state = np.matmul(total_operator_i, state)

        # Calculate expectation value
        rho = np.outer(state, state)
        sys = partial_trace(rho, [2, 18], axis=1)
        exp_sig = np.trace(np.matmul(sys, pauli_z))
        data.append(exp_sig.item())

    return data

# Rerun the calculations with these modifications
data_no_hermit = run_no_hermit()
data_hermit = run_hermit()

# Plot the results
fig, ax = plt.subplots(figsize=(10, 8))

# Plot individual datasets
ax.plot(steps, data_no_hermit, label="No Hermit Conjugate", linewidth=2)
ax.plot(steps, data_hermit, label="Hermit Conjugate", linewidth=2)

# Customize plot
ax.set_title("Evolution of System Expectation Values", fontsize=16)
ax.set_xlabel("Steps (collision with ith env state)", fontsize=14)
ax.set_ylabel("Expectation Value", fontsize=14)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(fontsize=12)
plt.show()
