from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import pec
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise
import matplotlib.pyplot as plt
import time
import os

# -----------------------------
# VQE ansatz circuit (2 qubits, no measurement yet - mitiq needs raw circuit)
# -----------------------------
def vqe_ansatz(theta):
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.ry(theta[2], 0)
    qc.ry(theta[3], 1)
    return qc

theta = [0.2, 0.3, 0.15, 0.25]
NOISE_LEVEL = 0.05

noise_model = NoiseModel()
error_1q = depolarizing_error(NOISE_LEVEL, 1)
error_2q = depolarizing_error(NOISE_LEVEL, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['ry'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

SHOTS = 4096

def z0_expectation(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        b0 = int(bits[-1])
        parity = 1 if b0 == 0 else -1
        exp += parity * count
    return exp / shots

def executor(circuit, noisy=True):
    qc = circuit.copy()
    qc.measure_all()
    backend = AerSimulator(noise_model=noise_model if noisy else None)
    job = backend.run(qc, shots=SHOTS)
    counts = job.result().get_counts()
    return z0_expectation(counts, SHOTS)

def noisy_executor(circuit):
    return executor(circuit, noisy=True)

qc = vqe_ansatz(theta)

representations = represent_operations_in_circuit_with_local_depolarizing_noise(
    ideal_circuit=qc,
    noise_level=NOISE_LEVEL,
)

ideal_value = executor(qc, noisy=False)
noisy_value = noisy_executor(qc)

print("Running PEC on VQE ansatz...")
start_time = time.time()

mitigated_value_raw = pec.execute_with_pec(
    qc,
    noisy_executor,
    representations=representations,
    num_samples=2000,
)

pec_runtime = time.time() - start_time
mitigated_value = max(-1.0, min(1.0, mitigated_value_raw))

error_before = abs(ideal_value - noisy_value)
error_after = abs(ideal_value - mitigated_value)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("\nIdeal Z0 expectation:      ", ideal_value)
print("Noisy Z0 expectation:      ", noisy_value)
print("PEC Mitigated (raw):       ", mitigated_value_raw)
print("PEC Mitigated (clipped):   ", mitigated_value)
print("Error before mitigation:   ", error_before)
print("Error after mitigation:    ", error_after)
print(f"Error reduction:            {reduction:.2f}%")
print(f"PEC runtime:                 {pec_runtime:.2f} seconds")

os.makedirs("results", exist_ok=True)
labels = ["Ideal", "Noisy", "PEC Mitigated"]
values = [ideal_value, noisy_value, mitigated_value]
colors = ["green", "red", "teal"]

plt.bar(labels, values, color=colors)
plt.ylabel("Z0 Expectation Value")
plt.title("VQE Ansatz: Ideal vs Noisy vs PEC-Mitigated")
plt.axhline(y=ideal_value, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/vqe_pec_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/vqe_pec_comparison.png")