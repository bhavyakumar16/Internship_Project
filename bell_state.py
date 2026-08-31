from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
# Create a circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Apply Hadamard gate
qc.h(0)

# Apply CNOT gate
qc.cx(0, 1)

# Measure both qubits
qc.measure([0, 1], [0, 1])

# Display the circuit
print(qc.draw())

# -----------------------------
# Run on Aer Simulator
# -----------------------------

simulator = AerSimulator()
job = simulator.run(qc, shots=1024)

result = job.result()
counts = result.get_counts()
print("\nMeasurement Counts:")
print(counts)