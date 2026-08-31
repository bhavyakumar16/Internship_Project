from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import numpy as np
import os

# -----------------------------
# QAOA circuit (2 qubits, p=1 layer)
# Max-Cut style: cost layer (RZZ via CX-RZ-CX) + mixer layer (RX)
# -----------------------------
def qaoa_circuit(gamma, beta):
    qc = QuantumCircuit(2, 2)

    # Initial superposition
    qc.h(0)
    qc.h(1)

    # Cost layer (encodes the "edge" between qubit 0 and 1)
    qc.cx(0, 1)
    qc.rz(2 * gamma, 1)
    qc.cx(0, 1)

    # Mixer layer
    qc.rx(2 * beta, 0)
    qc.rx(2 * beta, 1)

    qc.measure([0, 1], [0, 1])
    return qc

# Fixed angles (representing a converged/near-optimal QAOA parameter set)
gamma = np.pi / 4
beta = np.pi / 8

SHOTS = 8192

backend = AerSimulator()

qc = qaoa_circuit(gamma, beta)

print("Circuit:")
print(qc)

ideal_counts = backend.run(qc, shots=SHOTS).result().get_counts()

print("\nIdeal counts:", ideal_counts)

# -----------------------------
# Save histogram
# -----------------------------
os.makedirs("results", exist_ok=True)

plot_histogram(ideal_counts)
plt.title("QAOA - Ideal (No Noise)")
plt.savefig("results/qaoa_ideal.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved: results/qaoa_ideal.png")