from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import cdr
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# VQE ansatz circuit (2 qubits, no measurement yet - mitiq needs raw circuit)
# Small rotation angles keep the ideal state close to |00>, giving Z0 a
# strong, unambiguous ideal signal (near +1) instead of a weak near-zero one
# -----------------------------
def vqe_ansatz(theta):
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.ry(theta[2], 0)
    qc.ry(theta[3], 1)
    return qc

theta = [0.2, 0.3, 0.15, 0.25]  # small angles -> ideal state stays near |00>

NOISE_LEVEL = 0.05

# -----------------------------
# Noise model (standardized: 5% depolarizing, gates only)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(NOISE_LEVEL, 1)
error_2q = depolarizing_error(NOISE_LEVEL, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['ry'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

SHOTS = 8192

# -----------------------------
# Z0 expectation value
# -----------------------------
def z0_expectation(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        b0 = int(bits[-1])
        parity = 1 if b0 == 0 else -1
        exp += parity * count
    return exp / shots

def noisy_executor(circuit):
    qc = circuit.copy()
    qc.measure_all()
    backend = AerSimulator(noise_model=noise_model)
    job = backend.run(qc, shots=SHOTS)
    counts = job.result().get_counts()
    return z0_expectation(counts, SHOTS)

def ideal_executor(circuit):
    qc = circuit.copy()
    qc.measure_all()
    backend = AerSimulator()
    job = backend.run(qc, shots=SHOTS)
    counts = job.result().get_counts()
    return z0_expectation(counts, SHOTS)

# -----------------------------
# Run: Ideal vs Noisy vs CDR-Mitigated
# -----------------------------
qc = vqe_ansatz(theta)

ideal_value = ideal_executor(qc)
noisy_value = noisy_executor(qc)

print("Ideal Z0 (sanity check, should be well away from zero):", ideal_value)

print("\nRunning CDR on VQE ansatz (generating Clifford training circuits + fitting)...")

mitigated_value_raw = cdr.execute_with_cdr(
    qc,
    noisy_executor,
    simulator=ideal_executor,
    num_training_circuits=30,
)

mitigated_value = max(-1.0, min(1.0, mitigated_value_raw))

if mitigated_value_raw != mitigated_value:
    print(f"\n⚠ WARNING: raw CDR result ({mitigated_value_raw:.4f}) was outside [-1, 1], clipped.")

error_before = abs(ideal_value - noisy_value)
error_after = abs(ideal_value - mitigated_value)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("\nIdeal Z0 expectation:       ", ideal_value)
print("Noisy Z0 expectation:       ", noisy_value)
print("CDR Mitigated expectation:  ", mitigated_value)
print("Error before mitigation:    ", error_before)
print("Error after mitigation:     ", error_after)
print(f"Error reduction:             {reduction:.2f}%")
print("Training circuits used:      30")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "CDR Mitigated"]
values = [ideal_value, noisy_value, mitigated_value]
colors = ["green", "red", "brown"]

plt.bar(labels, values, color=colors)
plt.ylabel("Z0 Expectation Value")
plt.title("VQE Ansatz: Ideal vs Noisy vs CDR-Mitigated")
plt.axhline(y=ideal_value, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/vqe_cdr_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/vqe_cdr_comparison.png")