import numpy as np
from qutip import *
import matplotlib.pyplot as plt

############################################################################
# 1) HELPER FUNCTIONS
############################################################################

def fresh_ancilla(q=0.5, c=1.0):
    """
    Returns a qubit state with:
      - Probability q of |0>, (1-q) of |1>,
      - Off-diagonal coherence scaled by c * sqrt[q(1-q)].
    """
    rho_diag = (1 - q) * ket2dm(basis(2, 1)) + q * ket2dm(basis(2, 0))
    # Add coherence term
    coherence = c * np.sqrt(q*(1-q)) * (
        basis(2, 1)*basis(2, 0).dag() + basis(2, 0)*basis(2, 1).dag()
    )
    rho = rho_diag + coherence
    return rho / rho.tr()

def get_passive_energy(rho, levels):
    """
    Computes the 'passive energy' by:
      1) Sorting rho's eigenvalues descending
      2) Dotting with the ascending energy levels
    """
    eigvals, _ = rho.eigenstates()
    eigvals = np.array(eigvals, dtype=float)
    sorted_pop = np.sort(eigvals)[::-1]  # descending order
    return np.dot(sorted_pop, levels)

def partial_swap_map(rho, p):
    """"
    Applies the partial swap map to a 2-qubit density matrix rho.
    """
    S = swap()
    return (1 - p) * rho + p * S * rho * S


############################################################################
# 2) MAIN COLLISION SIMULATION
############################################################################

def run_collisions(
    N=10,               # Battery dimension is N+1
    collision_num=50,   # Number of collisions (battery-charger + battery-env)
    g=1.0,              # Jaynes–Cummings coupling strength
    theta=np.pi/4,      # Duration/angle for battery-charger collision
    q_charge=0.75,      # Probability for charger ancilla in |0>, etc.
    c_charge=1.0,       # Coherence scale for charger ancilla
    q_env=0.5,          # Probability for environment ancilla in |0>
    c_env=1.0,          # Coherence scale for environment ancilla
    kappa=0.5,          # Rate factor for battery-environment Hamiltonian
    delta_t=1.0,        # Time step for environment collision
    p=0.5               # Probability of applying environment collision vs. doing nothing
):
    """
    Repeated-collision model:

    Steps per collision:
      1) Battery + 'charger' qubit (fresh ancilla) evolve unitarily (Jaynes–Cummings style).
      2) Partial trace out charger, keep new battery state.
      3) Battery + environment qubit evolve via a *mixed channel*:
         (a) Build battery+environment
         (b) apply mixture: p * U_env(...) + (1-p)*Id
         (c) partial trace environment
      4) Measure battery's ergotropy
      5) Re-tensor battery with a fresh charger qubit for next collision.

    Returns:
      A list of ergotropy values after each collision.
    """
    #------------------------------
    # (A) Build battery operators
    #------------------------------
    dim_b = N + 1
    # Ladder ops for the battery
    A = sum(basis(dim_b,n-1)*basis(dim_b,n).dag() for n in range(1, dim_b))
    A_dag = A.dag()

    #------------------------------
    # (B) Charger collision operator
    #     H_charge = g * (A sigma^+ + A_dag sigma^-)
    #------------------------------
    V_charge = g * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U_charge = (-1j * theta * V_charge).expm()  # dimension (N+1)*2

    #------------------------------
    # (C) Environment collision operator (bare)
    #     H_env = sqrt(kappa/dt)*(A sigma^+ + A_dag sigma^-)
    #------------------------------
    V_env = np.sqrt(kappa / delta_t) * (tensor(A, sigmap()) + tensor(A_dag, sigmam()))
    U_env = (-1j * V_env).expm()

    #------------------------------
    # (D) Initialize battery in |0>
    #------------------------------
    battery = ket2dm(basis(dim_b, 0))  # dimension N+1
    # Combine with fresh charger ancilla:
    full_state = tensor(battery, fresh_ancilla(q_charge, c_charge))

    # We'll store ergotropy after each collision
    ergotropy_vals = []
    levels = np.arange(dim_b)  # energies = 0..N

    #------------------------------
    # (E) Collision Loop
    #------------------------------
    for col in range(1, collision_num + 1):
        # 1) Battery-Charger collision (unitary)
        full_state = U_charge * full_state * U_charge.dag()

        # 2) Extract battery alone
        battery = full_state.ptrace(0)

        # 3) Battery-Environment collision as a *mixed channel*
        env_state = fresh_ancilla(q_env, c_env)  # environment ancilla
        bat_env_in = tensor(battery, env_state)

        # Apply mixture: p * U_env(...) + (1-p)*Id
        bat_env_out = apply_mixture_channel(bat_env_in, U_env, p)

        # Partial trace environment to get battery
        battery = bat_env_out.ptrace(0)

        # 4) Measure ergotropy = (E - E_passive)
        probs = battery.diag()
        avg_energy = np.dot(levels, probs)
        passive_en = get_passive_energy(battery, levels)
        ergotropy = avg_energy - passive_en
        ergotropy_vals.append(ergotropy)

        # 5) Re-tensor battery with a *fresh charger* for next round
        full_state = tensor(battery, fresh_ancilla(q_charge, c_charge))

    return ergotropy_vals


############################################################################
# 3) DEMO / PLOT
############################################################################

if __name__ == "__main__":

    # Example usage:
    N = 10               # Battery dimension is 11
    collision_num = 50   # Fewer collisions for a quick demonstration
    p = 0.5              # Probability of environment collision
    kappa_values = [0.01, 0.2, 0.5, 1.0, 2.0]

    plt.figure()
    for kappa in kappa_values:
        erg_list = run_collisions(
            N=N, 
            collision_num=collision_num,
            kappa=kappa,
            p=p, 
            q_charge=0.75, 
            c_charge=1.0,
            q_env=0.5, 
            c_env=1.0
        )
        plt.plot(range(1, collision_num+1), erg_list, label=f"kappa={kappa}")

    plt.xlabel("Collision Number")
    plt.ylabel("Ergotropy")
    plt.title("Repeated Collision with Mixed (p) Environment Interaction")
    plt.legend()
    plt.show()
