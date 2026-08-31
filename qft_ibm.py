from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    SamplerV2
)

from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt


# =============================
# Connect IBM Quantum
# =============================

print("Connecting to IBM Quantum...")

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="drdo-internship"
)

print("Connected")

backend = service.backend("ibm_fez")

print("Backend:", backend.name)


# =============================
# Create QFT Circuit
# =============================

num_qubits = 3

qc = QuantumCircuit(num_qubits)

qft = QFT(
    num_qubits,
    do_swaps=True
)

qc.compose(
    qft,
    inplace=True
)

qc.measure_all()


print("\n========== QFT CIRCUIT ==========")
print(qc.draw())


print("\nNumber of Qubits:", qc.num_qubits)
print("Circuit Depth:", qc.depth())


# =============================
# Transpilation
# =============================

print("\nTranspiling...")

pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=1
)

transpiled_qc = pm.run(qc)


print("\n========== TRANSPILED CIRCUIT ==========")
print(transpiled_qc.draw())


# =============================
# Run on IBM Hardware
# =============================

shots = 1024

print("\nSubmitting job...")

sampler = SamplerV2(
    mode=backend
)


job = sampler.run(
    [transpiled_qc],
    shots=shots
)


print("Job ID:", job.job_id())


print("\nWaiting for result...")

result = job.result()


# =============================
# Counts
# =============================

pub_result = result[0]

counts = pub_result.data.meas.get_counts()


print("\n========== QFT COUNTS ==========")
print(counts)


# =============================
# Histogram
# =============================

plt.figure(figsize=(7,5))

plot_histogram(
    counts
)

plt.title("QFT - IBM Quantum Hardware")

plt.savefig(
    "qft_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =============================
# Experiment Summary
# =============================

print("\n========== EXPERIMENT SUMMARY ==========")

print("Backend:", backend.name)

print("Number of Qubits:", qc.num_qubits)

print("Shots:", shots)

print("Original Circuit Depth:", qc.depth())

print(
    "Transpiled Circuit Depth:",
    transpiled_qc.depth()
)

print("Histogram saved as qft_histogram.png")