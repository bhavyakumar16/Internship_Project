from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
import numpy as np
import matplotlib.pyplot as plt
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
# Noise model: standardized 5% depolarizing (gates) + 5% readout error
# (readout error included specifically because MEM targets measurement errors)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(0.05, 1)
error_2q = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

p0given1 = 0.05
p1given0 = 0.05
readout_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
noise_model.add_all_qubit_readout_error(readout_error)

backend = AerSimulator(noise_model=noise_model)
ideal_backend = AerSimulator()

# -----------------------------
# Step 1: Ideal result
# -----------------------------
qc = ghz_circuit()
ideal_counts = ideal_backend.run(qc, shots=SHOTS).result().get_counts()

# -----------------------------
# Step 2: Noisy result (before mitigation)
# -----------------------------
noisy_counts = backend.run(qc, shots=SHOTS).result().get_counts()

# -----------------------------
# Step 3: Calibration - measure all 8 possible basis states to build confusion matrix
# NOTE: reversed(bitstring) because Qiskit's bit ordering is qubit 0 = rightmost
# -----------------------------
def calibration_circuit(bitstring):
    qc = QuantumCircuit(3, 3)
    for i, bit in enumerate(reversed(bitstring)):
        if bit == '1':
            qc.x(i)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc

basis_states = ['000', '001', '010', '011', '100', '101', '110', '111']
state_to_idx = {s: i for i, s in enumerate(basis_states)}
cal_matrix = np.zeros((8, 8))

for prep_state in basis_states:
    cal_qc = calibration_circuit(prep_state)
    cal_counts = backend.run(cal_qc, shots=SHOTS).result().get_counts()
    prep_idx = state_to_idx[prep_state]
    for measured_state, count in cal_counts.items():
        measured_clean = measured_state.replace(" ", "")
        meas_idx = state_to_idx[measured_clean]
        cal_matrix[meas_idx][prep_idx] += count / SHOTS

print("Calibration (confusion) matrix built (8x8).")

# -----------------------------
# Step 4: Invert the calibration matrix and apply to noisy counts
# -----------------------------
mitigation_matrix = np.linalg.inv(cal_matrix)

noisy_vector = np.zeros(8)
for state, count in noisy_counts.items():
    clean_state = state.replace(" ", "")
    noisy_vector[state_to_idx[clean_state]] = count / SHOTS

mitigated_vector = mitigation_matrix @ noisy_vector
mitigated_vector = np.clip(mitigated_vector, 0, None)
mitigated_vector = mitigated_vector / mitigated_vector.sum()

mitigated_counts = {
    state: mitigated_vector[i] * SHOTS
    for i, state in enumerate(basis_states)
}

# -----------------------------
# Compare using ZZZ expectation values
# -----------------------------
def zzz_expectation_from_counts(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        if bits == '000' or bits == '111':
            exp += count
        else:
            exp -= count
    return exp / shots

ideal_zzz = zzz_expectation_from_counts(ideal_counts, SHOTS)
noisy_zzz = zzz_expectation_from_counts(noisy_counts, SHOTS)
mitigated_zzz = zzz_expectation_from_counts(mitigated_counts, SHOTS)

error_before = abs(ideal_zzz - noisy_zzz)
error_after = abs(ideal_zzz - mitigated_zzz)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("\nIdeal counts:    ", ideal_counts)
print("Noisy counts:    ", noisy_counts)
print("Mitigated counts:", {k: round(v, 1) for k, v in mitigated_counts.items()})

print(f"\nIdeal ZZZ expectation:      {ideal_zzz:.4f}")
print(f"Noisy ZZZ expectation:      {noisy_zzz:.4f}")
print(f"MEM Mitigated expectation:  {mitigated_zzz:.4f}")
print(f"Error before mitigation:    {error_before:.4f}")
print(f"Error after mitigation:     {error_after:.4f}")
print(f"Error reduction:             {reduction:.2f}%")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "MEM Mitigated"]
values = [ideal_zzz, noisy_zzz, mitigated_zzz]
colors = ["green", "red", "purple"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZZ Expectation Value")
plt.title("GHZ State: Ideal vs Noisy vs MEM-Mitigated")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/ghz_mem_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/ghz_mem_comparison.png")