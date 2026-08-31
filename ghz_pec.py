from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import pec
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise
import matplotlib.pyplot as plt
import time
import os

# -----------------------------
# GHZ circuit (3 qubits, no measurement yet - mitiq needs raw circuit)
# -----------------------------
def ghz_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc

NOISE_LEVEL = 0.05

# -----------------------------
# Noise model (standardized: 5% depolarizing, gates only - no readout for PEC)
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
# Executor
# -----------------------------
def executor(circuit, noisy=True):
    qc = circuit.copy()
    qc.measure_all()
    backend = AerSimulator(noise_model=noise_model if noisy else None)
    job = backend.run(qc, shots=SHOTS)
    counts = job.result().get_counts()
    return compute_zzz_expectation(counts, SHOTS)

def noisy_executor(circuit):
    return executor(circuit, noisy=True)

# -----------------------------
# Build PEC noise representations
# -----------------------------
qc = ghz_circuit()

representations = represent_operations_in_circuit_with_local_depolarizing_noise(
    ideal_circuit=qc,
    noise_level=NOISE_LEVEL,
)

# -----------------------------
# Run: Ideal vs Noisy vs PEC-Mitigated
# -----------------------------
ideal_value = executor(qc, noisy=False)
noisy_value = noisy_executor(qc)

print("Running PEC on GHZ (this samples many circuit variants, may take a bit)...")
start_time = time.time()

mitigated_value = pec.execute_with_pec(
    qc,
    noisy_executor,
    representations=representations,
    num_samples=8000,
)

pec_runtime = time.time() - start_time

error_before = abs(ideal_value - noisy_value)
error_after = abs(ideal_value - mitigated_value)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("\nIdeal ZZZ expectation:      ", ideal_value)
print("Noisy ZZZ expectation:      ", noisy_value)
print("PEC Mitigated expectation:  ", mitigated_value)
print("Error before mitigation:    ", error_before)
print("Error after mitigation:     ", error_after)
print(f"Error reduction:             {reduction:.2f}%")
print(f"PEC runtime:                  {pec_runtime:.2f} seconds")
print(f"Sampling cost:                8000 circuit samples")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "PEC Mitigated"]
values = [ideal_value, noisy_value, mitigated_value]
colors = ["green", "red", "teal"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZZ Expectation Value")
plt.title("GHZ State: Ideal vs Noisy vs PEC-Mitigated")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/ghz_pec_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/ghz_pec_comparison.png")