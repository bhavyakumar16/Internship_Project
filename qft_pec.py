from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import pec
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise
import matplotlib.pyplot as plt
import time
import os

# -----------------------------
# QFT circuit: QFT followed by inverse QFT (ideal = Identity -> |001>, Z0 = -1)
# -----------------------------
def qft_circuit():
    qc = QuantumCircuit(3)
    qc.x(0)
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
# Executor
# -----------------------------
def executor(circuit, noisy=True):
    qc = circuit.copy()
    qc.measure_all()
    backend = AerSimulator(noise_model=noise_model if noisy else None)
    job = backend.run(qc, shots=SHOTS)
    counts = job.result().get_counts()
    return z0_expectation(counts, SHOTS)

def noisy_executor(circuit):
    return executor(circuit, noisy=True)

# -----------------------------
# Build PEC noise representations
# -----------------------------
qc = qft_circuit()

representations = represent_operations_in_circuit_with_local_depolarizing_noise(
    ideal_circuit=qc,
    noise_level=NOISE_LEVEL,
)

# -----------------------------
# Run: Ideal vs Noisy vs PEC-Mitigated
# -----------------------------
ideal_value = executor(qc, noisy=False)
noisy_value = noisy_executor(qc)

print("Running PEC on QFT+inverse-QFT (this samples many circuit variants, may take a bit)...")
start_time = time.time()

mitigated_value = pec.execute_with_pec(
    qc,
    noisy_executor,
    representations=representations,
    num_samples=2000,
)

pec_runtime = time.time() - start_time

error_before = abs(ideal_value - noisy_value)
error_after = abs(ideal_value - mitigated_value)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("\nIdeal Z0 expectation:       ", ideal_value)
print("Noisy Z0 expectation:       ", noisy_value)
print("PEC Mitigated expectation:  ", mitigated_value)
print("Error before mitigation:    ", error_before)
print("Error after mitigation:     ", error_after)
print(f"Error reduction:             {reduction:.2f}%")
print(f"PEC runtime:                  {pec_runtime:.2f} seconds")
print(f"Sampling cost:                2000 circuit samples")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "PEC Mitigated"]
values = [ideal_value, noisy_value, mitigated_value]
colors = ["green", "red", "teal"]

plt.bar(labels, values, color=colors)
plt.ylabel("Z0 Expectation Value")
plt.title("QFT + inverse-QFT: Ideal vs Noisy vs PEC-Mitigated")
plt.axhline(y=ideal_value, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/qft_pec_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/qft_pec_comparison.png")