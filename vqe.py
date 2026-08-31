from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator

import matplotlib.pyplot as plt

# -----------------------------
# Hamiltonian
# -----------------------------
hamiltonian = SparsePauliOp.from_list([
    ("ZZ", -1),
    ("XX", 1)
])

# -----------------------------
# Ansatz
# -----------------------------
ansatz = TwoLocal(2, "ry", "cz", reps=1)

# -----------------------------
# Optimizer
# -----------------------------
optimizer = COBYLA(maxiter=50)

# -----------------------------
# Store energies
# -----------------------------
energies = []

def callback(eval_count, params, energy, metadata):
    energies.append(energy)

# -----------------------------
# Estimator
# -----------------------------
estimator = StatevectorEstimator()

# -----------------------------
# VQE
# -----------------------------
vqe = VQE(
    estimator,
    ansatz,
    optimizer,
    callback=callback
)

result = vqe.compute_minimum_eigenvalue(hamiltonian)

print("\n========== VQE RESULT ==========")
print("Ground State Energy:", result.eigenvalue.real)

print("\n========== CIRCUIT ==========")
print(ansatz.draw())

print("\nCircuit Depth:", ansatz.depth())
print("Number of Qubits:", ansatz.num_qubits)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(range(1, len(energies)+1), energies, marker='o')
plt.title("VQE Energy Convergence")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.grid(True)

plt.savefig("vqe_convergence.png", dpi=300)

print("\nGraph saved as vqe_convergence.png")