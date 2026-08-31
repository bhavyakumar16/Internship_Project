from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import zne
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import LinearFactory
import matplotlib.pyplot as plt
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

# -----------------------------
# Noise model (standardized: 5% depolarizing)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(0.05, 1)
error_2q = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

SHOTS = 8192

# -----------------------------
# Compute ZZZ expectation value from counts
# Correct GHZ outcomes are '000' or '111' -> +1
# Any other outcome (leakage from noise) -> -1
# (Perfect GHZ state gives +1. Noise pulls it toward 0.)
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
# Run: Ideal vs Noisy vs ZNE-Mitigated
# -----------------------------
qc = ghz_circuit()

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

print("Ideal ZZZ expectation:      ", ideal_value)
print("Noisy ZZZ expectation:      ", noisy_value)
print("ZNE Mitigated expectation:  ", mitigated_value)
print("Error before mitigation:    ", error_before)
print("Error after mitigation:     ", error_after)
print(f"Error reduction:             {reduction:.2f}%")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "ZNE Mitigated"]
values = [ideal_value, noisy_value, mitigated_value]
colors = ["green", "red", "blue"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZZ Expectation Value")
plt.title("GHZ State: Ideal vs Noisy vs ZNE-Mitigated")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/ghz_zne_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/ghz_zne_comparison.png")