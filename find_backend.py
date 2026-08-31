from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="drdo-internship"
)

backends = [
    "ibm_brisbane",
    "ibm_sherbrooke",
    "ibm_kyiv",
    "ibm_fez",
    "ibm_marrakesh",
    "ibm_torino",
    "ibm_nazca"
]

for name in backends:
    try:
        backend = service.backend(name)
        print("AVAILABLE:", name)
    except Exception:
        print("NOT AVAILABLE:", name)