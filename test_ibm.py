from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum_platform")

print("Account loaded!")

instances = service.instances()

print("Instances:")
print(instances)