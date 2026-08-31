from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import numpy as np
import os

# -----------------------------
# Simple VQE-style ansatz circuit (2 qubits)
# Hardware-efficient ansatz: RY rotations + entangling CX
# -----------------------------
def vqe_ansatz(theta):
    qc = QuantumCircuit(2, 2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.ry(theta[2], 0)
    qc.ry(theta[3], 1)
    qc.measure([0, 1], [0, 1])
    return qc

# Fixed set of angles (representing a converged/near-optimal VQE parameter set)
theta = [np.pi/4, np.pi/3, np.pi/6, np.pi/2]

SHOTS = 8192

backend = AerSimulator()

qc = vqe_ansatz(theta)

print("Circuit:")
print(qc)

ideal_counts = backend.run(qc, shots=SHOTS).result().get_counts()

print("\nIdeal counts:", ideal_counts)

# -----------------------------
# Save histogram
# -----------------------------
os.makedirs("results", exist_ok=True)

plot_histogram(ideal_counts)
plt.title("VQE Ansatz - Ideal (No Noise)")
plt.savefig("results/vqe_ideal.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved: results/vqe_ideal.png")