from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt


# -----------------------------
# IBM Quantum Connection
# -----------------------------
print("Connecting to IBM Quantum...")

service = QiskitRuntimeService()

print("Connected!")


# -----------------------------
# Backend
# -----------------------------
backend = service.backend("ibm_fez")

print("Backend:", backend.name)



# -----------------------------
# Bell State Circuit
# -----------------------------
qc = QuantumCircuit(2)

qc.h(0)
qc.cx(0, 1)

qc.measure_all()


print("\n========== ORIGINAL CIRCUIT ==========")
print(qc)



# -----------------------------
# Transpilation
# -----------------------------
print("\nTranspiling...")

pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=1
)

transpiled_qc = pm.run(qc)


print("\n========== TRANSPILED CIRCUIT ==========")
print(transpiled_qc)


print("\nCircuit Depth:", transpiled_qc.depth())
print("Number of Qubits:", transpiled_qc.num_qubits)



# -----------------------------
# Run on IBM QPU
# -----------------------------
print("\nSubmitting job...")

sampler = SamplerV2(
    mode=backend
)

job = sampler.run(
    [transpiled_qc],
    shots=1024
)


print("Job ID:", job.job_id())



# -----------------------------
# Get Result
# -----------------------------
print("\nWaiting for result...")

result = job.result()


counts = result[0].data.meas.get_counts()



print("\n========== RESULT ==========")
print(counts)



# -----------------------------
# Histogram
# -----------------------------
print("\nCreating Histogram...")

plot_histogram(
    counts
)

plt.title("Bell State Results on IBM Quantum Hardware")
plt.xlabel("Measurement States")
plt.ylabel("Counts")

plt.savefig(
    "bell_ibm_histogram.png",
    dpi=300,
    bbox_inches="tight"
)


print("Histogram saved as bell_ibm_histogram.png")


plt.show()