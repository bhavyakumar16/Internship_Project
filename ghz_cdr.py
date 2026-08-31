from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import cdr
import matplotlib.pyplot as plt
import os

# -----------------------------
# GHZ circuit (3 qubits)
# -----------------------------
def ghz_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc

NOISE_LEVEL = 0.05

# -----------------------------
# Noise model (standardized: 5% depolarizing, gates only)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(NOISE_LEVEL, 1)
error_2q = depolarizing_error(NOISE_LEVEL, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

SHOTS = 4096

# -----------------------------
# ZZZ expectation value helper
# -----------------------------
def compute_zzz_expectation(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        if bits == '000' or bits == '111':
            exp += count
        else:
            exp -= count
    return exp / shots

# -----------------------------
# Noisy executor (used both for real circuit and training circuits)
# -----------------------------
def noisy_executor(circuit):
    qc = circuit.copy()
    qc.measure_all()
    backend = AerSimulator(noise_model=noise_model)
    job = backend.run(qc, shots=SHOTS)
    counts = job.result().get_counts()
    return compute_zzz_expectation(counts, SHOTS)

# -----------------------------
# Exact/ideal executor (noiseless simulation - used for training circuits only)
# -----------------------------
def ideal_executor(circuit):
    qc = circuit.copy()
    qc.measure_all()
    backend = AerSimulator()
    job = backend.run(qc, shots=SHOTS)
    counts = job.result().get_counts()
    return compute_zzz_expectation(counts, SHOTS)

# -----------------------------
# Run: Ideal vs Noisy vs CDR-Mitigated
# -----------------------------
qc = ghz_circuit()

ideal_value = ideal_executor(qc)
noisy_value = noisy_executor(qc)

print("Running CDR on GHZ (generating Clifford training circuits + fitting)...")

mitigated_value = cdr.execute_with_cdr(
    qc,
    noisy_executor,
    simulator=ideal_executor,
    num_training_circuits=15,
)

error_before = abs(ideal_value - noisy_value)
error_after = abs(ideal_value - mitigated_value)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("\nIdeal ZZZ expectation:      ", ideal_value)
print("Noisy ZZZ expectation:      ", noisy_value)
print("CDR Mitigated expectation:  ", mitigated_value)
print("Error before mitigation:    ", error_before)
print("Error after mitigation:     ", error_after)
print(f"Error reduction:             {reduction:.2f}%")
print("Training circuits used:      15")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "CDR Mitigated"]
values = [ideal_value, noisy_value, mitigated_value]
colors = ["green", "red", "brown"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZZ Expectation Value")
plt.title("GHZ State: Ideal vs Noisy vs CDR-Mitigated")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/ghz_cdr_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/ghz_cdr_comparison.png")