from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import numpy as np
import matplotlib.pyplot as plt
import os

NOISE_LEVEL = 0.05
SHOTS = 8192
gamma = np.pi / 4
beta = np.pi / 8

noise_model = NoiseModel()
error_1q = depolarizing_error(NOISE_LEVEL, 1)
error_2q = depolarizing_error(NOISE_LEVEL, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rz', 'rx'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# -----------------------------
# QAOA circuit builder (unmeasured, reusable for both copies)
# -----------------------------
def qaoa_ops(gamma, beta, qubits):
    def build(qc):
        qc.h(qubits[0])
        qc.h(qubits[1])
        qc.cx(qubits[0], qubits[1])
        qc.rz(2 * gamma, qubits[1])
        qc.cx(qubits[0], qubits[1])
        qc.rx(2 * beta, qubits[0])
        qc.rx(2 * beta, qubits[1])
    return build

def full_circuit():
    qc = QuantumCircuit(2, 2)
    qaoa_ops(gamma, beta, [0, 1])(qc)
    qc.measure([0, 1], [0, 1])
    return qc

# -----------------------------
# Joint ZZ parity (matches Bell/GHZ/VQE VD methodology)
# -----------------------------
def zz_expectation(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        b0, b1 = int(bits[-1]), int(bits[-2])
        parity = 1 if (b0 + b1) % 2 == 0 else -1
        exp += parity * count
    return exp / shots

ideal_backend = AerSimulator()
noisy_backend = AerSimulator(noise_model=noise_model)

qc = full_circuit()
ideal_counts = ideal_backend.run(qc, shots=SHOTS).result().get_counts()
noisy_counts = noisy_backend.run(qc, shots=SHOTS).result().get_counts()

ideal_val = zz_expectation(ideal_counts, SHOTS)
noisy_val = zz_expectation(noisy_counts, SHOTS)

# -----------------------------
# Virtual Distillation circuit (M=2 copies)
# Qubits 0,1 = Copy A. Qubits 2,3 = Copy B.
# -----------------------------
vd_qc = QuantumCircuit(4, 4)
qaoa_ops(gamma, beta, [0, 1])(vd_qc)
qaoa_ops(gamma, beta, [2, 3])(vd_qc)

vd_qc.cx(2, 0)
vd_qc.cx(3, 1)
vd_qc.h(2)
vd_qc.h(3)

vd_qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

vd_counts = noisy_backend.run(vd_qc, shots=SHOTS).result().get_counts()

numerator = 0
denominator = 0

for bitstring, count in vd_counts.items():
    bits = bitstring.replace(" ", "")
    a0, a1 = int(bits[-1]), int(bits[-2])
    b0, b1 = int(bits[-3]), int(bits[-4])

    copyA_parity = 1 if (a0 + a1) % 2 == 0 else -1
    copyB_parity = 1 if (b0 + b1) % 2 == 0 else -1

    numerator += copyA_parity * copyB_parity * count
    denominator += copyB_parity * count

vd_mitigated_val_raw = numerator / denominator if denominator != 0 else 0
vd_mitigated_val = max(-1.0, min(1.0, vd_mitigated_val_raw))

error_before = abs(ideal_val - noisy_val)
error_after = abs(ideal_val - vd_mitigated_val)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("Ideal ZZ expectation:        ", ideal_val)
print("Noisy ZZ expectation:        ", noisy_val)
print("VD Mitigated (raw):          ", vd_mitigated_val_raw)
print("VD Mitigated (clipped):      ", vd_mitigated_val)
print("Error before mitigation:     ", error_before)
print("Error after mitigation:      ", error_after)
print(f"Error reduction:              {reduction:.2f}%")

os.makedirs("results", exist_ok=True)
labels = ["Ideal", "Noisy", "VD Mitigated"]
values = [ideal_val, noisy_val, vd_mitigated_val]
colors = ["green", "red", "orchid"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZ Expectation Value")
plt.title("QAOA: Ideal vs Noisy vs VD-Mitigated")
plt.axhline(y=ideal_val, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/qaoa_vd_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/qaoa_vd_comparison.png")