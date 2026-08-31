from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
# Create 3-qubit circuit
qc = QuantumCircuit(3, 3)
# Prepare state |001>
qc.x(0)
print("Initial Circuit:")
print(qc)
# Apply QFT
qft = QFT(num_qubits=3)
qc.append(qft, [0, 1, 2])
# IMPORTANT: Decompose QFT into basic gates
qc = qc.decompose(reps=10)
# Measure
qc.measure([0,1,2],[0,1,2])
print("\nQFT Circuit:")
print(qc)
# Simulator
simulator = AerSimulator()

job = simulator.run(qc, shots=1024)
result = job.result()

counts = result.get_counts()

print("\nMeasurement Counts:")
print(counts)

# Histogram
plot_histogram(counts)
plt.savefig("qft_histogram.png")
print("\nHistogram saved as qft_histogram.png")

# Parameters
print("\nCircuit Depth:", qc.depth())
print("Number of Qubits:", qc.num_qubits)
print("Shots:", 1024)