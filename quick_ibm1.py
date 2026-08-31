from qiskit_ibm_runtime import QiskitRuntimeService

print("Connecting...")

service = QiskitRuntimeService()

print("Connected")

backend = service.backend("ibm_fez")

print("Backend loaded:", backend.name)