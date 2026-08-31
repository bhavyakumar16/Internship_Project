from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


# --------------------------------
# Connect IBM Quantum
# --------------------------------

print("Connecting to IBM Quantum...")

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="drdo-internship"
)

print("Connected")


# --------------------------------
# Backend
# --------------------------------

backend = service.backend("ibm_fez")

print("Backend:", backend.name)



# --------------------------------
# GHZ Circuit
# --------------------------------

qc = QuantumCircuit(3)

# GHZ state preparation
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# Measurement
qc.measure_all()


print("\n========== ORIGINAL CIRCUIT ==========")
print(qc.draw())


# Circuit information

num_qubits = qc.num_qubits
depth = qc.depth()
shots = 1024


print("\nNumber of Qubits:", num_qubits)
print("Circuit Depth:", depth)



# --------------------------------
# Transpilation
# --------------------------------

print("\nTranspiling...")


pm = generate_preset_pass_manager(
    optimization_level=1,
    backend=backend
)


transpiled = pm.run(qc)


print("\n========== TRANSPILED CIRCUIT ==========")
print(transpiled.draw())



# --------------------------------
# Run on IBM QPU
# --------------------------------

print("\nSubmitting job...")


sampler = SamplerV2(
    mode=backend
)


job = sampler.run(
    [transpiled],
    shots=shots
)


print("Job ID:", job.job_id())


print("\nWaiting for result...")


result = job.result()



# --------------------------------
# Counts
# --------------------------------

counts = result[0].data.meas.get_counts()


print("\n========== GHZ COUNTS ==========")
print(counts)



# --------------------------------
# Histogram
# --------------------------------

plt.figure(figsize=(7,5))


plot_histogram(
    counts
)


plt.title(
    "GHZ State Histogram - IBM QPU"
)


plt.xlabel("Measurement States")
plt.ylabel("Counts")


plt.savefig(
    "ghz_histogram.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print("\nHistogram saved as ghz_histogram.png")



# --------------------------------
# Benchmark Summary
# --------------------------------

print("\n========== BENCHMARK SUMMARY ==========")

print("Circuit Name      : GHZ State Generation")
print("Backend           :", backend.name)
print("Number of Qubits  :", num_qubits)
print("Shots             :", shots)
print("Circuit Depth     :", depth)
print("Job ID            :", job.job_id())

print("\nMeasurement Counts:")
print(counts)

print("======================================")