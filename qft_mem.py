from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# QFT circuit (3 qubits)
# -----------------------------
def qft_circuit():
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.append(QFT(num_qubits=3, do_swaps=True), [0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    qc = transpile(qc, basis_gates=['h', 'x', 'cx', 'cp', 'swap', 'rz', 'measure'], optimization_level=0)
    return qc

SHOTS = 8192

# -----------------------------
# Noise model: standardized 5% depolarizing + 5% readout error (for MEM)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(0.05, 1)
error_2q = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cp', 'swap'])

p0given1 = 0.05
p1given0 = 0.05
readout_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
noise_model.add_all_qubit_readout_error(readout_error)

backend = AerSimulator(noise_model=noise_model)
ideal_backend = AerSimulator()

# -----------------------------
# Step 1: Ideal result
# -----------------------------
qc = qft_circuit()
ideal_counts = ideal_backend.run(qc, shots=SHOTS).result().get_counts()

# -----------------------------
# Step 2: Noisy result (before mitigation)
# -----------------------------
noisy_counts = backend.run(qc, shots=SHOTS).result().get_counts()

# -----------------------------
# Step 3: Calibration - measure all 8 possible basis states to build confusion matrix
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
# Compare using classical fidelity to ideal distribution
# -----------------------------
def get_probs(counts, shots_or_total):
    probs = {s: 0.0 for s in basis_states}
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        probs[bits] = count / shots_or_total
    return probs

def fidelity(probs_a, probs_b):
    total = 0
    for s in basis_states:
        total += np.sqrt(max(probs_a[s], 0) * max(probs_b[s], 0))
    return total ** 2

ideal_probs = get_probs(ideal_counts, SHOTS)
noisy_probs = get_probs(noisy_counts, SHOTS)
mitigated_probs = get_probs(mitigated_counts, SHOTS)

ideal_fid = 1.0
noisy_fid = fidelity(ideal_probs, noisy_probs)
mitigated_fid = fidelity(ideal_probs, mitigated_probs)

error_before = abs(ideal_fid - noisy_fid)
error_after = abs(ideal_fid - mitigated_fid)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("\nIdeal counts:    ", ideal_counts)
print("Noisy counts:    ", noisy_counts)
print("Mitigated counts:", {k: round(v, 1) for k, v in mitigated_counts.items()})

print(f"\nIdeal fidelity:      {ideal_fid:.4f}")
print(f"Noisy fidelity:      {noisy_fid:.4f}")
print(f"MEM Mitigated fidelity: {mitigated_fid:.4f}")
print(f"Error before mitigation: {error_before:.4f}")
print(f"Error after mitigation:  {error_after:.4f}")
print(f"Error reduction:          {reduction:.2f}%")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "MEM Mitigated"]
values = [ideal_fid, noisy_fid, mitigated_fid]
colors = ["green", "red", "purple"]

plt.bar(labels, values, color=colors)
plt.ylabel("Fidelity to Ideal Distribution")
plt.title("QFT: Ideal vs Noisy vs MEM-Mitigated")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/qft_mem_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/qft_mem_comparison.png")