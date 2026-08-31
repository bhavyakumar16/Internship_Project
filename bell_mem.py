from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import os

# -----------------------------
# Bell circuit
# -----------------------------
def bell_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

SHOTS = 8192

# -----------------------------
# Noise model: gate noise + readout error combined
# (realistic — real hardware has both)
# -----------------------------
noise_model = NoiseModel()

error_1q = depolarizing_error(0.02, 1)
error_2q = depolarizing_error(0.02, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

p0given1 = 0.05
p1given0 = 0.05
readout_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
noise_model.add_all_qubit_readout_error(readout_error)

backend = AerSimulator(noise_model=noise_model)
ideal_backend = AerSimulator()

# -----------------------------
# Step 1: Ideal result (reference)
# -----------------------------
qc = bell_circuit()
ideal_counts = ideal_backend.run(qc, shots=SHOTS).result().get_counts()

# -----------------------------
# Step 2: Noisy result (before mitigation)
# -----------------------------
noisy_counts = backend.run(qc, shots=SHOTS).result().get_counts()

# -----------------------------
# Step 3: Calibration — measure known basis states to build confusion matrix
# -----------------------------
def calibration_circuit(bitstring):
    qc = QuantumCircuit(2, 2)
    for i, bit in enumerate(bitstring):
        if bit == '1':
            qc.x(i)
    qc.measure([0, 1], [0, 1])
    return qc

basis_states = ['00', '01', '10', '11']
cal_matrix = np.zeros((4, 4))

state_to_idx = {s: i for i, s in enumerate(basis_states)}

for prep_state in basis_states:
    cal_qc = calibration_circuit(prep_state)
    cal_counts = backend.run(cal_qc, shots=SHOTS).result().get_counts()
    prep_idx = state_to_idx[prep_state]
    for measured_state, count in cal_counts.items():
        measured_clean = measured_state.replace(" ", "")
        meas_idx = state_to_idx[measured_clean]
        cal_matrix[meas_idx][prep_idx] += count / SHOTS

print("Calibration (confusion) matrix:")
print(cal_matrix)

# -----------------------------
# Step 4: Invert the calibration matrix and apply to noisy counts
# -----------------------------
mitigation_matrix = np.linalg.inv(cal_matrix)

noisy_vector = np.zeros(4)
for state, count in noisy_counts.items():
    clean_state = state.replace(" ", "")
    noisy_vector[state_to_idx[clean_state]] = count / SHOTS

mitigated_vector = mitigation_matrix @ noisy_vector
mitigated_vector = np.clip(mitigated_vector, 0, None)  # remove tiny negative artifacts
mitigated_vector = mitigated_vector / mitigated_vector.sum()  # renormalize to valid probabilities

mitigated_counts = {
    state: mitigated_vector[i] * SHOTS
    for i, state in enumerate(basis_states)
}

# -----------------------------
# Compare using ZZ expectation values
# -----------------------------
def zz_expectation_from_counts(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        parity = 1 if bits.count('1') % 2 == 0 else -1
        exp += parity * count
    return exp / shots

ideal_zz = zz_expectation_from_counts(ideal_counts, SHOTS)
noisy_zz = zz_expectation_from_counts(noisy_counts, SHOTS)
mitigated_zz = zz_expectation_from_counts(mitigated_counts, SHOTS)

error_before = abs(ideal_zz - noisy_zz)
error_after = abs(ideal_zz - mitigated_zz)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("\nIdeal counts:    ", ideal_counts)
print("Noisy counts:    ", noisy_counts)
print("Mitigated counts:", {k: round(v, 1) for k, v in mitigated_counts.items()})

print(f"\nIdeal ZZ expectation:      {ideal_zz:.4f}")
print(f"Noisy ZZ expectation:      {noisy_zz:.4f}")
print(f"MEM Mitigated expectation: {mitigated_zz:.4f}")
print(f"Error before mitigation:   {error_before:.4f}")
print(f"Error after mitigation:    {error_after:.4f}")
print(f"Error reduction:            {reduction:.2f}%")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "MEM Mitigated"]
values = [ideal_zz, noisy_zz, mitigated_zz]
colors = ["green", "red", "purple"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZ Expectation Value")
plt.title("Bell State: Ideal vs Noisy vs MEM-Mitigated")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/bell_mem_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/bell_mem_comparison.png")