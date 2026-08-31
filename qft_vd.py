from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt
import os

NOISE_LEVEL = 0.05
SHOTS = 8192

# -----------------------------
# Noise model (standardized: 5% depolarizing)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(NOISE_LEVEL, 1)
error_2q = depolarizing_error(NOISE_LEVEL, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'rz'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# -----------------------------
# QFT + inverse QFT circuit builder (3 qubits, unmeasured)
# -----------------------------
def qft_inv_qft_ops():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.append(QFT(num_qubits=3, do_swaps=True), [0, 1, 2])
    qc.append(QFT(num_qubits=3, do_swaps=True).inverse(), [0, 1, 2])
    qc = transpile(qc, basis_gates=['h', 'x', 'cx', 'rz'], optimization_level=0)
    return qc

# -----------------------------
# Standard noisy circuit (for ideal/noisy baseline comparison)
# -----------------------------
def full_circuit():
    qc = QuantumCircuit(3, 3)
    qc.compose(qft_inv_qft_ops(), [0, 1, 2], inplace=True)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc

def z0_expectation(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        b0 = int(bits[-1])
        parity = 1 if b0 == 0 else -1
        exp += parity * count
    return exp / shots

ideal_backend = AerSimulator()
noisy_backend = AerSimulator(noise_model=noise_model)

qc = full_circuit()
ideal_counts = ideal_backend.run(qc, shots=SHOTS).result().get_counts()
noisy_counts = noisy_backend.run(qc, shots=SHOTS).result().get_counts()

ideal_val = z0_expectation(ideal_counts, SHOTS)
noisy_val = z0_expectation(noisy_counts, SHOTS)

# -----------------------------
# Virtual Distillation circuit (M=2 copies)
# Qubits 0,1,2 = Copy A. Qubits 3,4,5 = Copy B.
# -----------------------------
vd_qc = QuantumCircuit(6, 6)

vd_qc.compose(qft_inv_qft_ops(), [0, 1, 2], inplace=True)   # Copy A
vd_qc.compose(qft_inv_qft_ops(), [3, 4, 5], inplace=True)   # Copy B

# Derangement circuit: entangle corresponding qubits across copies
vd_qc.cx(3, 0)
vd_qc.cx(4, 1)
vd_qc.cx(5, 2)
vd_qc.h(3)
vd_qc.h(4)
vd_qc.h(5)

vd_qc.measure([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])

vd_counts = noisy_backend.run(vd_qc, shots=SHOTS).result().get_counts()

# -----------------------------
# Apply VD formula: Z0 on Copy A, weighted by joint parity of Copy B (derangement register)
# -----------------------------
numerator = 0
denominator = 0

for bitstring, count in vd_counts.items():
    bits = bitstring.replace(" ", "")
    # bits[-1]=q0 (Copy A, qubit 0); bits[-4]=q3, bits[-5]=q4, bits[-6]=q5 (Copy B)
    a0 = int(bits[-1])
    b0, b1, b2 = int(bits[-4]), int(bits[-5]), int(bits[-6])

    observable_A = 1 if a0 == 0 else -1
    derangement_parity = 1 if (b0 + b1 + b2) % 2 == 0 else -1

    numerator += observable_A * derangement_parity * count
    denominator += derangement_parity * count

vd_mitigated_val = numerator / denominator if denominator != 0 else 0

# -----------------------------
# Results
# -----------------------------
error_before = abs(ideal_val - noisy_val)
error_after = abs(ideal_val - vd_mitigated_val)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("Ideal Z0 expectation:        ", ideal_val)
print("Noisy Z0 expectation:        ", noisy_val)
print("VD Mitigated expectation:    ", vd_mitigated_val)
print("Error before mitigation:     ", error_before)
print("Error after mitigation:      ", error_after)
print(f"Error reduction:              {reduction:.2f}%")
print("Extra qubit cost:              2x (6 qubits used instead of 3)")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "VD Mitigated"]
values = [ideal_val, noisy_val, vd_mitigated_val]
colors = ["green", "red", "orchid"]

plt.bar(labels, values, color=colors)
plt.ylabel("Z0 Expectation Value")
plt.title("QFT + inverse-QFT: Ideal vs Noisy vs VD-Mitigated")
plt.axhline(y=ideal_val, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/qft_vd_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/qft_vd_comparison.png")