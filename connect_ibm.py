from qiskit_ibm_runtime import QiskitRuntimeService

print("Connecting to IBM Quantum...")

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="drdo-internship"
)

print("Connected!")

backend = service.backend(
    "ibm_fez",
    use_fractional_gates=False
)

print("Backend:", backend.name)
print("Qubits:", backend.num_qubits)