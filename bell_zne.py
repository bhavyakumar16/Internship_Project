from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import zne
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory, LinearFactory
import matplotlib.pyplot as plt
import os

# -----------------------------
# Bell circuit (no measurement yet - mitiq needs raw circuit)
# -----------------------------
def bell_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

# -----------------------------
# Noise model (same depolarizing noise as before)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(0.05, 1)
error_2q = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

SHOTS = 8192  # more shots = less statistical noise in the expectation value

# -----------------------------
# Compute ZZ expectation value from counts
# (Perfect Bell state gives +1. Noise pulls it toward 0.)
# -----------------------------
def compute_zz_expectation(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        parity = 1 if bits.count('1') % 2 == 0 else -1
        exp += parity * count
    return exp / shots

# -----------------------------
# Executor: runs circuit, returns expectation value
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
# Run: Ideal vs Noisy
# -----------------------------
qc = bell_circuit()

ideal_value = executor(qc, noisy=False)
noisy_value = noisy_executor(qc)

# -----------------------------
# ZNE with global folding + Richardson extrapolation
# (more stable than local folding on short circuits)
# -----------------------------
mitigated_richardson = zne.execute_with_zne(
    qc,
    noisy_executor,
    scale_noise=fold_global,
    factory=RichardsonFactory(scale_factors=[1, 2, 3])
)

# -----------------------------
# ZNE with global folding + Linear extrapolation
# (simpler, sometimes more robust than Richardson on very short circuits)
# -----------------------------
mitigated_linear = zne.execute_with_zne(
    qc,
    noisy_executor,
    scale_noise=fold_global,
    factory=LinearFactory(scale_factors=[1, 2, 3])
)

def report(name, value):
    error_before = abs(ideal_value - noisy_value)
    error_after = abs(ideal_value - value)
    reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0
    print(f"\n{name}")
    print(f"  Mitigated value:  {value:.4f}")
    print(f"  Error before:     {error_before:.4f}")
    print(f"  Error after:      {error_after:.4f}")
    print(f"  Error reduction:  {reduction:.2f}%")
    return reduction

print("Ideal ZZ expectation:", ideal_value)
print("Noisy ZZ expectation:", noisy_value)

red_richardson = report("ZNE (Global Fold + Richardson)", mitigated_richardson)
red_linear = report("ZNE (Global Fold + Linear)", mitigated_linear)

# -----------------------------
# Save comparison bar chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "ZNE\n(Richardson)", "ZNE\n(Linear)"]
values = [ideal_value, noisy_value, mitigated_richardson, mitigated_linear]
colors = ["green", "red", "blue", "orange"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZ Expectation Value")
plt.title("Bell State: Ideal vs Noisy vs ZNE-Mitigated")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/bell_zne_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/bell_zne_comparison.png")