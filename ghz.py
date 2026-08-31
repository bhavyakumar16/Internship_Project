import matplotlib
matplotlib.use("Agg")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# ----------------------------
# Create GHZ Circuit
# ----------------------------
qc = QuantumCircuit(3)

qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

qc.measure_all()

print("Circuit:\n")
print(qc.draw())

# ----------------------------
# Run on Simulator
# ----------------------------
simulator = AerSimulator()

job = simulator.run(qc, shots=1024)

result = job.result()

counts = result.get_counts()

print("\nMeasurement Counts:")
print(counts)

# ----------------------------
# Histogram
# ----------------------------
fig = plot_histogram(counts)

# Save histogram as image
fig.savefig("ghz_histogram.png")

# Show histogram
plt.show()

print("\nHistogram saved as ghz_histogram.png")

print("\nCircuit Depth:", qc.depth())
print("Number of Qubits:", qc.num_qubits)
print("Shots:", 1024)