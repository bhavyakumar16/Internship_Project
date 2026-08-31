from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import os

# -----------------------------
# QFT circuit (3 qubits)
# Input: |001> (a simple, non-trivial basis state)
# -----------------------------
def qft_circuit():
    qc = QuantumCircuit(3, 3)
    qc.x(0)  # prepare input state |001>
    qc.append(QFT(num_qubits=3, do_swaps=True), [0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc

SHOTS = 8192

# -----------------------------
# Noise model (standardized: 5% depolarizing)
# Note: QFT uses additional gates (h, cp/crz, swap) beyond h/cx
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(0.05, 1)
error_2q = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cp', 'swap'])

ideal_backend = AerSimulator()
noisy_backend = AerSimulator(noise_model=noise_model)

qc = qft_circuit()
qc = transpile(qc, basis_gates=['h', 'x', 'cx', 'cp', 'swap', 'rz'], optimization_level=0)

print("Circuit:")
print(qc)

ideal_counts = ideal_backend.run(qc, shots=SHOTS).result().get_counts()
noisy_counts = noisy_backend.run(qc, shots=SHOTS).result().get_counts()

print("\nIdeal counts:", ideal_counts)
print("Noisy counts:", noisy_counts)

# -----------------------------
# Save histograms
# -----------------------------
os.makedirs("results", exist_ok=True)

plot_histogram(ideal_counts)
plt.title("QFT - Ideal (No Noise)")
plt.savefig("results/qft_ideal.png", dpi=300, bbox_inches="tight")
plt.close()

plot_histogram(noisy_counts)
plt.title("QFT - Noisy (Depolarizing)")
plt.savefig("results/qft_noisy.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved: results/qft_ideal.png, results/qft_noisy.png")