from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError,
    coherent_unitary_error,
)
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import os

# -----------------------------
# Ideal Bell circuit
# -----------------------------
def bell_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

qc = bell_circuit()

# -----------------------------
# Helper to run and save
# -----------------------------
def run_and_save(circuit, noise_model, filename, title):
    sim = AerSimulator(noise_model=noise_model)
    job = sim.run(circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()
    print(f"\n{title}: {counts}")

    plot_histogram(counts)
    plt.title(title)
    plt.savefig(f"results/{filename}", dpi=300, bbox_inches="tight")
    plt.close()
    return counts

os.makedirs("results", exist_ok=True)

# -----------------------------
# 1. Ideal (no noise)
# -----------------------------
run_and_save(qc, None, "bell_ideal.png", "Bell - Ideal (No Noise)")

# -----------------------------
# 2. Depolarizing noise
# -----------------------------
noise_depol = NoiseModel()
error_1q = depolarizing_error(0.05, 1)
error_2q = depolarizing_error(0.05, 2)
noise_depol.add_all_qubit_quantum_error(error_1q, ['h'])
noise_depol.add_all_qubit_quantum_error(error_2q, ['cx'])
run_and_save(qc, noise_depol, "bell_depolarizing.png", "Bell - Depolarizing Noise")

# -----------------------------
# 3. Amplitude damping
# -----------------------------
noise_amp = NoiseModel()
error_amp = amplitude_damping_error(0.05)
noise_amp.add_all_qubit_quantum_error(error_amp, ['h'])
noise_amp.add_all_qubit_quantum_error(error_amp.tensor(error_amp), ['cx'])
run_and_save(qc, noise_amp, "bell_amplitude_damping.png", "Bell - Amplitude Damping")

# -----------------------------
# 4. Phase damping
# -----------------------------
noise_phase = NoiseModel()
error_phase = phase_damping_error(0.05)
noise_phase.add_all_qubit_quantum_error(error_phase, ['h'])
noise_phase.add_all_qubit_quantum_error(error_phase.tensor(error_phase), ['cx'])
run_and_save(qc, noise_phase, "bell_phase_damping.png", "Bell - Phase Damping")

# -----------------------------
# 5. Readout error
# -----------------------------
noise_readout = NoiseModel()
p0given1 = 0.05
p1given0 = 0.05
readout_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
noise_readout.add_all_qubit_readout_error(readout_error)
run_and_save(qc, noise_readout, "bell_readout.png", "Bell - Readout Error")

# -----------------------------
# 6. Coherent gate error
# -----------------------------
theta = 0.1  # small rotation error
coherent_unitary = np.array([
    [np.cos(theta / 2), -np.sin(theta / 2)],
    [np.sin(theta / 2), np.cos(theta / 2)]
])
noise_coherent = NoiseModel()
error_coherent = coherent_unitary_error(coherent_unitary)
noise_coherent.add_all_qubit_quantum_error(error_coherent, ['h'])
run_and_save(qc, noise_coherent, "bell_coherent.png", "Bell - Coherent Gate Error")

print("\nAll 6 experiments complete. Check the 'results/' folder.")