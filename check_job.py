from qiskit_ibm_runtime import QiskitRuntimeService

print("Connecting...")

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="drdo-internship"
)

print("Connected")

job_id = "d9b0gvnu62s738011eg"

job = service.retrieve_job(job_id)

print("\nJob ID:", job_id)

print("Status:", job.status())