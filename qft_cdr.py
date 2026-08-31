from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import cdr
import matplotlib.pyplot as plt
import os

# -----------------------------
# QFT circuit: apply QFT then inverse QFT (QFT† · QFT = Identity)
# Ideal output returns exactly to input state |001>, giving a strong,
# non-degenerate Z0 signal while still exercising the full noisy gate set
# -----------------------------
def qft_circuit():
    qc = QuantumCircuit(3)
    qc.x(0)  # prepare |001>
    qc.append(QFT(num_qubits=3, do_swaps=True), [0, 1, 2])
    qc.append(QFT(num_qubits=3, do_swaps=True).inverse(), [0, 1, 2])
    qc = transpile(qc, basis_gates=['h', 'x', 'cx', 'rz'], optimization_level=0)
    return qc

NOISE_LEVEL = 0.05

# -----------------------------
# Noise model (standardized: 5% depolarizing, gates only)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(NOISE_LEVEL, 1)
error_2q = depolarizing_error(NOISE_LEVEL, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'rz'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

SHOTS = 4096

# -----------------------------
# Z0 expectation value
# Ideal circuit = Identity, so ideal output is exactly |001> -> Z0 = -1
# -----------------------------
def z0_expectation(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        b0 = int(bits[-1])
        parity = 1 if b0 == 0 else -1
        exp += parity * count
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
    return z0_expectation(counts, SHOTS)

# -----------------------------
# Exact/ideal executor (noiseless simulation - used for training circuits only)
# -----------------------------
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
qc = qft_circuit()

ideal_value = ideal_executor(qc)
noisy_value = noisy_executor(qc)

print("Running CDR on QFT + inverse-QFT (generating Clifford training circuits + fitting)...")

mitigated_value = cdr.execute_with_cdr(
    qc,
    noisy_executor,
    simulator=ideal_executor,
    num_training_circuits=15,
)

error_before = abs(ideal_value - noisy_value)
error_after = abs(ideal_value - mitigated_value)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("\nIdeal Z0 expectation:       ", ideal_value)
print("Noisy Z0 expectation:       ", noisy_value)
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
plt.ylabel("Z0 Expectation Value")
plt.title("QFT + inverse-QFT: Ideal vs Noisy vs CDR-Mitigated")
plt.axhline(y=ideal_value, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/qft_cdr_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/qft_cdr_comparison.png")