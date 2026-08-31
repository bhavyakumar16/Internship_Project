from qiskit_ibm_runtime import QiskitRuntimeService

print("Connecting...")

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="drdo-internship"
)

print("Connected!")

backend = service.backend("ibm_marrakesh")
print("Backend:", backend.name)