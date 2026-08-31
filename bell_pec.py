from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import pec
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise
import matplotlib.pyplot as plt
import time
import os

# -----------------------------
# Bell circuit
# -----------------------------
def bell_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

NOISE_LEVEL = 0.02  # keep this small - PEC math assumes fairly light noise

# -----------------------------
# Noise model
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(NOISE_LEVEL, 1)
error_2q = depolarizing_error(NOISE_LEVEL, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

SHOTS = 4096

# -----------------------------
# ZZ expectation value helper
# -----------------------------
def compute_zz_expectation(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        parity = 1 if bits.count('1') % 2 == 0 else -1
        exp += parity * count
    return exp / shots

# -----------------------------
# Executor
# -----------------------------
def executor(circuit, noisy=True):
    qc = circuit.copy()
    qc.measure_all()
    backend = AerSimulator(noise_model=noise_model if noisy else None)
    job = backend.run(qc, shots=SHOTS)
    counts = job.result().get_counts()
    return compute_zz_expectation(counts, SHOTS)

def noisy_executor(circuit):
    return executor(circuit, noisy=True)

# -----------------------------
# Build PEC noise representations
# (tells Mitiq how to "invert" each gate's depolarizing noise)
# -----------------------------
qc = bell_circuit()

representations = represent_operations_in_circuit_with_local_depolarizing_noise(
    ideal_circuit=qc,
    noise_level=NOISE_LEVEL,
)

# -----------------------------
# Run: Ideal vs Noisy vs PEC-Mitigated
# -----------------------------
ideal_value = executor(qc, noisy=False)
noisy_value = noisy_executor(qc)

print("Running PEC (this samples many circuit variants, may take a bit)...")
start_time = time.time()

mitigated_value = pec.execute_with_pec(
    qc,
    noisy_executor,
    representations=representations,
    num_samples=2000,  # increased from 200 for more stable/accurate estimate
)

pec_runtime = time.time() - start_time

error_before = abs(ideal_value - noisy_value)
error_after = abs(ideal_value - mitigated_value)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("\nIdeal ZZ expectation:      ", ideal_value)
print("Noisy ZZ expectation:      ", noisy_value)
print("PEC Mitigated expectation: ", mitigated_value)
print("Error before mitigation:   ", error_before)
print("Error after mitigation:    ", error_after)
print(f"Error reduction:            {reduction:.2f}%")
print(f"PEC runtime:                 {pec_runtime:.2f} seconds")
print(f"Sampling cost:                2000 circuit samples (vs 1 for ZNE base run)")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "PEC Mitigated"]
values = [ideal_value, noisy_value, mitigated_value]
colors = ["green", "red", "teal"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZ Expectation Value")
plt.title("Bell State: Ideal vs Noisy vs PEC-Mitigated")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/bell_pec_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/bell_pec_comparison.png")