from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA

import matplotlib.pyplot as plt

# -----------------------------
# Cost Hamiltonian
# -----------------------------
hamiltonian = SparsePauliOp.from_list([
    ("ZZ", 1),
    ("ZI", -1),
    ("IZ", -1)
])

# -----------------------------
# Optimizer
# -----------------------------
optimizer = COBYLA(maxiter=50)

# -----------------------------
# Sampler
# -----------------------------
sampler = StatevectorSampler()

# -----------------------------
# Store optimization values
# -----------------------------
energies = []

def callback(eval_count, params, energy, metadata):
    energies.append(energy)

# -----------------------------
# QAOA
# -----------------------------
qaoa = QAOA(
    sampler=sampler,
    optimizer=optimizer,
    reps=2,
    callback=callback
)

result = qaoa.compute_minimum_eigenvalue(hamiltonian)

print("\n========== QAOA RESULT ==========")
print("Minimum Energy:", result.eigenvalue.real)

print("\nCircuit Depth:", qaoa.ansatz.decompose().depth())
print("Number of Qubits:", qaoa.ansatz.num_qubits)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(range(1, len(energies)+1), energies, marker='o')
plt.title("QAOA Energy Convergence")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.grid(True)

plt.savefig("qaoa_convergence.png", dpi=300)

print("\nGraph saved as qaoa_convergence.png")