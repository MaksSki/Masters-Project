# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:02:26 2024

@author: rithi
"""

import numpy as np
import matplotlib.pyplot as plt

def partial_trace(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])

def random_unitary(seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    # Generate a random complex matrix
    random_matrix = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    
    # Perform QR decomposition and normalize to get a unitary matrix
    q, _ = np.linalg.qr(random_matrix)
    return q

#pauli matrices
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0,1],[1,0]])
pauli_z = np.array([[1,0],[0,-1]])

#initial condition
dt = 0.1
U_op = random_unitary()
p = 0.8

zs = np.array([1,0])
os = np.array([0,1])

A = np.array([[np.sqrt(1-p),np.sqrt(p)]
              ,[0,0]
              ,[0,0]
              ,[np.sqrt(p),np.sqrt(1-p)]])

len_states = 10

state = np.kron(zs,zs) #2nd is our system

def run_model_no_uncompute(state,U_op):
    #data
    data = []
    steps = np.arange(1,len_states+1,step=1)
    
    for i in range(len_states):
        if i == 0:
            state_next = np.matmul(np.kron(A,np.eye(2)),state)
            state = state_next
            
            operator = np.kron(np.kron(U_op,np.eye(2)),U_op)
            
            state = np.matmul(operator,state)

        else:
            state_next = np.matmul(np.kron(np.kron(np.eye(2**i),A),np.eye(2)),state)
            state = state_next

            operator = np.kron(np.kron(np.eye(2**i),U_op),np.kron(np.eye(2),U_op))
            
            state = np.matmul(operator,state)

        rho = np.outer(state,state)
        sys = partial_trace(rho, [2**(i+2),2],axis=0) #sys
        exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
        data.append(exp_sig.item())

    return data,steps

def run_model_uncompute(state,U_op):
    #data
    data = []
    steps = np.arange(1,len_states+1,step=1)
    
    for i in range(len_states):
        if i == 0:
            state_next = np.matmul(np.kron(A,np.eye(2)),state)
            state = state_next
            
            operator = np.kron(np.kron(U_op,np.eye(2)),U_op)
            
            state = np.matmul(operator,state)
            memory = state
            
            #uncompute
            state = state.reshape(2,4)
            U,S,Vh = np.linalg.svd(state,full_matrices=False)
            
            U_dag = np.conjugate(U).T
            
            state = np.dot(np.dot(U_dag,U),np.dot(np.diag(S),Vh))
            state = state.flatten()

            rho = np.outer(state,state)
            sys = partial_trace(rho, [2**(i+2),2],axis=0) #sys
            exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
            data.append(exp_sig.item())
            
                        
            #test
            rho_mem = np.outer(memory,memory)
            rho_state = np.outer(state, state)
            
            c1 = partial_trace(rho_mem, [2,4],axis=0)
            c2 = partial_trace(rho_state, [2,4],axis=0)
            
            print(np.allclose(c1,c2))

        else:
            if len(state) == 16:
                state_next = np.matmul(np.kron(np.kron(np.eye(4),A),np.eye(2)),state)
                state = state_next

                # Step-by-step evaluation of the Kronecker product chain
                operator_part_1 = np.kron(np.eye(4), U_op)  # 4x4 identity with U
                operator_part_2 = np.kron(np.eye(2), U_op)  # 2x2 identity with U
                
                # Combine to form the final operator
                operator = np.kron(operator_part_1, operator_part_2)

                state = np.matmul(operator,state)
                memory = state

                #uncompute
                state = state.reshape(8,4)
                U,S,Vh = np.linalg.svd(state,full_matrices=False)
                
                U_dag = np.conjugate(U).T
                
                state = np.dot(np.dot(U_dag,U),np.dot(np.diag(S),Vh))
                state = state.flatten()
                
                
                rho = np.outer(state,state)
                sys = partial_trace(rho, [8,2],axis=0) #sys
                exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
                data.append(exp_sig.item())
                
                            
                #test
                rho_mem = np.outer(memory,memory)
                rho_state = np.outer(state, state)
                
                c1 = partial_trace(rho_mem, [4,4],axis=0)
                c2 = partial_trace(rho_state, [4,4],axis=0)
                
                print(np.allclose(c1,c2))
            else:
                state_next = np.matmul(np.kron(np.kron(np.eye(2),A),np.eye(2)),state)
                state = state_next

                # Step-by-step evaluation of the Kronecker product chain
                operator_part_1 = np.kron(np.eye(2), U_op)  # 4x4 identity with U
                operator_part_2 = np.kron(np.eye(2), U_op)  # 2x2 identity with U
                
                # Combine to form the final operator
                operator = np.kron(operator_part_1, operator_part_2)

                state = np.matmul(operator,state)
                memory = state

                #uncompute
                state = state.reshape(4,4)
                U,S,Vh = np.linalg.svd(state,full_matrices=False)
                
                U_dag = np.conjugate(U).T
                
                state = np.dot(np.dot(U_dag,U),np.dot(np.diag(S),Vh))
                state = state.flatten()
                
                
                rho = np.outer(state,state)
                sys = partial_trace(rho, [8,2],axis=0) #sys
                exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
                data.append(exp_sig.item())
                
                            
                #test
                rho_mem = np.outer(memory,memory)
                rho_state = np.outer(state, state)
                
                c1 = partial_trace(rho_mem, [4,4],axis=0)
                c2 = partial_trace(rho_state, [4,4],axis=0)
                
                print(np.allclose(c1,c2))
                
    return data,steps
    
data,steps = run_model_no_uncompute(state,U_op)
data1,steps1 = run_model_uncompute(state, U_op)

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(steps, data, label="Expectation pauli z", color="blue", linewidth=2)
ax.plot(steps1, data1, label="Expectation pauli z with uncompute", color="red", linewidth=2)

# Add title and labels with proper font size
ax.set_title("Evolution of system", fontsize=16, pad=15)
ax.set_xlabel("Steps (collision with ith env state)", fontsize=14)
ax.set_ylabel("Expectation value", fontsize=14)

# Add a grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend with a frame
ax.legend(loc='upper right', fontsize=12, frameon=True)

# Increase the ticks font size for better visibility
ax.tick_params(axis='both', which='major', labelsize=12)

# Set tighter layout for padding
plt.tight_layout()

#plt.savefig("Evolution of expectation pauli z.jpg",dpi=200)
# Show the plot
plt.show()        
    