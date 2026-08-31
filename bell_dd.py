from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit.circuit.library import XGate
import matplotlib.pyplot as plt
import os

SHOTS = 8192

# -----------------------------
# Noise model: gate noise + idle-time decoherence (thermal relaxation)
# DD specifically fights this kind of idle-time noise
# -----------------------------
noise_model = NoiseModel()

error_1q = depolarizing_error(0.02, 1)
error_2q = depolarizing_error(0.02, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# T1/T2 relaxation during idle time (in ns) - simulates real decoherence while a qubit waits
t1, t2 = 50000, 30000
idle_error = thermal_relaxation_error(t1, t2, 5000)  # 5000ns idle delay
noise_model.add_quantum_error(idle_error, ['id'], [0])
noise_model.add_quantum_error(idle_error, ['id'], [1])

backend = AerSimulator(noise_model=noise_model)
ideal_backend = AerSimulator()

def zz_expectation_from_counts(counts, shots):
    exp = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        parity = 1 if bits.count('1') % 2 == 0 else -1
        exp += parity * count
    return exp / shots

# -----------------------------
# Bell circuit WITH a deliberate idle gap
# (qubits sit idle before measurement - simulated with delay)
# -----------------------------
def bell_with_idle():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.delay(5000, 0, unit='ns')  # qubit 0 sits idle
    qc.delay(5000, 1, unit='ns')  # qubit 1 sits idle
    qc.measure([0, 1], [0, 1])
    return qc

qc = bell_with_idle()

# -----------------------------
# Ideal (no noise) reference
# -----------------------------
ideal_counts = ideal_backend.run(qc, shots=SHOTS).result().get_counts()
ideal_zz = zz_expectation_from_counts(ideal_counts, SHOTS)

# -----------------------------
# Noisy, WITHOUT dynamical decoupling
# -----------------------------
noisy_counts = backend.run(qc, shots=SHOTS).result().get_counts()
noisy_zz = zz_expectation_from_counts(noisy_counts, SHOTS)

# -----------------------------
# Apply Dynamical Decoupling (X-X pulse sequence during idle periods)
# dt=1e-9 tells Qiskit that all durations below are in nanoseconds
# -----------------------------
durations = InstructionDurations(
    [("h", None, 50), ("cx", None, 300), ("x", None, 50), ("measure", None, 1000)],
    dt=1e-9
)

dd_sequence = [XGate(), XGate()]

pm = PassManager([
    ALAPScheduleAnalysis(durations),
    PadDynamicalDecoupling(durations, dd_sequence),
])

qc_dd = pm.run(qc)

dd_counts = backend.run(qc_dd, shots=SHOTS).result().get_counts()
dd_zz = zz_expectation_from_counts(dd_counts, SHOTS)

# -----------------------------
# Results
# -----------------------------
error_before = abs(ideal_zz - noisy_zz)
error_after = abs(ideal_zz - dd_zz)
reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0

print("Ideal ZZ expectation:         ", ideal_zz)
print("Noisy ZZ expectation (no DD): ", noisy_zz)
print("DD-Protected expectation:     ", dd_zz)
print("Error before DD:              ", error_before)
print("Error after DD:               ", error_after)
print(f"Error reduction:               {reduction:.2f}%")

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy\n(no DD)", "DD\nProtected"]
values = [ideal_zz, noisy_zz, dd_zz]
colors = ["green", "red", "steelblue"]

plt.bar(labels, values, color=colors)
plt.ylabel("ZZ Expectation Value")
plt.title("Bell State: Ideal vs Noisy vs Dynamical Decoupling")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/bell_dd_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/bell_dd_comparison.png")