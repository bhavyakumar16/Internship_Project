from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import os

# -----------------------------
# GHZ circuit (3 qubits)
# -----------------------------
def ghz_circuit():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc

SHOTS = 8192

# -----------------------------
# Noise model: depolarizing on gates (same style as Bell)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(0.05, 1)
error_2q = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

ideal_backend = AerSimulator()
noisy_backend = AerSimulator(noise_model=noise_model)

qc = ghz_circuit()

ideal_counts = ideal_backend.run(qc, shots=SHOTS).result().get_counts()
noisy_counts = noisy_backend.run(qc, shots=SHOTS).result().get_counts()

print("Ideal counts:", ideal_counts)
print("Noisy counts:", noisy_counts)

# -----------------------------
# Save histograms
# -----------------------------
os.makedirs("results", exist_ok=True)

plot_histogram(ideal_counts)
plt.title("GHZ - Ideal (No Noise)")
plt.savefig("results/ghz_ideal.png", dpi=300, bbox_inches="tight")
plt.close()

plot_histogram(noisy_counts)
plt.title("GHZ - Noisy (Depolarizing)")
plt.savefig("results/ghz_noisy.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved: results/ghz_ideal.png, results/ghz_noisy.png")