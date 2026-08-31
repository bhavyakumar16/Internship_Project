from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import zne
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import LinearFactory
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# QAOA circuit (2 qubits, p=1, no measurement yet - mitiq needs raw circuit)
# -----------------------------
def qaoa_circuit(gamma, beta):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.rz(2 * gamma, 1)
    qc.cx(0, 1)
    qc.rx(2 * beta, 0)
    qc.rx(2 * beta, 1)
    return qc

gamma = np.pi / 4
beta = np.pi / 8

# -----------------------------
# Noise model (standardized: 5% depolarizing)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(0.05, 1)
error_2q = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rz', 'rx'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

SHOTS = 8192

def zz_expectation(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        b0, b1 = int(bits[-1]), int(bits[-2])
        parity = 1 if (b0 + b1) % 2 == 0 else -1
        exp += parity * count
    return exp / shots

def executor(circuit, noisy=True):
    qc = circuit.copy()
    qc.measure_all()
    backend = AerSimulator(noise_model=noise_model if noisy else None)
    job = backend.run(qc, shots=SHOTS)
    counts = job.result().get_counts()
    return zz_expectation(counts, SHOTS)

def noisy_executor(circuit):
    return executor(circuit, noisy=True)

qc = qaoa_circuit(gamma, beta)

ideal_value = executor(qc, noisy=False)
noisy_value = noisy_executor(qc)

mitigated_value = zne.execute_with_zne(
    qc,
    noisy_executor,
    scale_noise=fold_global,
    factory=LinearFactory(scale_factors=[1, 2, 3])
)

error_before = abs(ideal_value - noisy_value)
error_after = abs(ideal_value - mitigated_value)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("Ideal ZZ expectation:      ", ideal_value)
print("Noisy ZZ expectation:      ", noisy_value)
print("ZNE Mitigated expectation: ", mitigated_value)
print("Error before mitigation:   ", error_before)
print("Error after mitigation:    ", error_after)
print(f"Error reduction:            {reduction:.2f}%")

os.makedirs("results", exist_ok=True)
labels = ["Ideal", "Noisy", "ZNE Mitigated"]
values = [ideal_value, noisy_value, mitigated_value]
colors = ["green", "red", "magenta"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZ Expectation Value")
plt.title("QAOA: Ideal vs Noisy vs ZNE-Mitigated")
plt.axhline(y=ideal_value, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/qaoa_zne_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/qaoa_zne_comparison.png")