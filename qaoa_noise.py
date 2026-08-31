from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import numpy as np
import os

# -----------------------------
# QAOA circuit (2 qubits, p=1 layer)
# -----------------------------
def qaoa_circuit(gamma, beta):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.rz(2 * gamma, 1)
    qc.cx(0, 1)
    qc.rx(2 * beta, 0)
    qc.rx(2 * beta, 1)
    qc.measure([0, 1], [0, 1])
    return qc

gamma = np.pi / 4
beta = np.pi / 8

SHOTS = 8192

# -----------------------------
# Noise model (standardized: 5% depolarizing)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(0.05, 1)
error_2q = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rz', 'rx'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

ideal_backend = AerSimulator()
noisy_backend = AerSimulator(noise_model=noise_model)

qc = qaoa_circuit(gamma, beta)

ideal_counts = ideal_backend.run(qc, shots=SHOTS).result().get_counts()
noisy_counts = noisy_backend.run(qc, shots=SHOTS).result().get_counts()

print("Ideal counts:", ideal_counts)
print("Noisy counts:", noisy_counts)

# -----------------------------
# Save histograms
# -----------------------------
os.makedirs("results", exist_ok=True)

plot_histogram(ideal_counts)
plt.title("QAOA - Ideal (No Noise)")
plt.savefig("results/qaoa_ideal.png", dpi=300, bbox_inches="tight")
plt.close()

plot_histogram(noisy_counts)
plt.title("QAOA - Noisy (Depolarizing)")
plt.savefig("results/qaoa_noisy.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved: results/qaoa_ideal.png, results/qaoa_noisy.png")