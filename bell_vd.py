from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt
import os

NOISE_LEVEL = 0.05
SHOTS = 8192

# -----------------------------
# Noise model
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(NOISE_LEVEL, 1)
error_2q = depolarizing_error(NOISE_LEVEL, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# -----------------------------
# Standard noisy Bell (for ideal/noisy baseline comparison)
# -----------------------------
def bell_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def zz_expectation_from_counts(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        parity = 1 if bits.count('1') % 2 == 0 else -1
        exp += parity * count
    return exp / shots

ideal_backend = AerSimulator()
noisy_backend = AerSimulator(noise_model=noise_model)

qc = bell_circuit()
ideal_counts = ideal_backend.run(qc, shots=SHOTS).result().get_counts()
noisy_counts = noisy_backend.run(qc, shots=SHOTS).result().get_counts()

ideal_zz = zz_expectation_from_counts(ideal_counts, SHOTS)
noisy_zz = zz_expectation_from_counts(noisy_counts, SHOTS)

# -----------------------------
# Virtual Distillation circuit (M=2 copies)
# Qubits 0,1 = Copy A. Qubits 2,3 = Copy B.
# -----------------------------
vd_qc = QuantumCircuit(4, 4)

# Prepare Copy A (Bell state)
vd_qc.h(0)
vd_qc.cx(0, 1)

# Prepare Copy B (independent noisy Bell state)
vd_qc.h(2)
vd_qc.cx(2, 3)

# Derangement circuit: entangle corresponding qubits across copies
vd_qc.cx(2, 0)
vd_qc.cx(3, 1)
vd_qc.h(2)
vd_qc.h(3)

vd_qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

vd_counts = noisy_backend.run(vd_qc, shots=SHOTS).result().get_counts()

# -----------------------------
# Apply VD formula:
# <Z0 Z1>_VD = E[(-1)^(b0+b1) * (-1)^(c0+c1)] / E[(-1)^(c0+c1)]
# where b0,b1 = Copy A bits (qubits 0,1), c0,c1 = Copy B bits (qubits 2,3)
# -----------------------------
numerator = 0
denominator = 0

for bitstring, count in vd_counts.items():
    bits = bitstring.replace(" ", "")
    # Qiskit bitstrings are reversed (qubit 0 = rightmost)
    b0, b1, c0, c1 = int(bits[-1]), int(bits[-2]), int(bits[-3]), int(bits[-4])

    copyA_parity = (-1) ** (b0 + b1)
    copyB_parity = (-1) ** (c0 + c1)

    numerator += copyA_parity * copyB_parity * count
    denominator += copyB_parity * count

vd_mitigated_zz = numerator / denominator if denominator != 0 else 0

# -----------------------------
# Results
# -----------------------------
error_before = abs(ideal_zz - noisy_zz)
error_after = abs(ideal_zz - vd_mitigated_zz)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("Ideal ZZ expectation:       ", ideal_zz)
print("Noisy ZZ expectation:       ", noisy_zz)
print("VD Mitigated expectation:   ", vd_mitigated_zz)
print("Error before mitigation:    ", error_before)
print("Error after mitigation:     ", error_after)
print(f"Error reduction:             {reduction:.2f}%")
print("Extra qubit cost:             2x (4 qubits used instead of 2)")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "VD Mitigated"]
values = [ideal_zz, noisy_zz, vd_mitigated_zz]
colors = ["green", "red", "orchid"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZ Expectation Value")
plt.title("Bell State: Ideal vs Noisy vs VD-Mitigated")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/bell_vd_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/bell_vd_comparison.png")