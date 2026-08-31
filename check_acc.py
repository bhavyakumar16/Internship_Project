from qiskit_ibm_runtime import QiskitRuntimeService

print("Checking account...")

service = QiskitRuntimeService()

print("Account connected!")

print("Available instances:")
for inst in service.instances():
    print(inst)