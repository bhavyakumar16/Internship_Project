from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="drdo-internship"
)

backend = service.backend("ibm_fez")

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.measure_all()

print("Starting transpile")

pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=1
)

new_qc = pm.run(qc)

print("DONE")
print(new_qc)