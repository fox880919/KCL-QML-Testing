import pennylane as qml
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

backend = service.backend("ibm_brisbane")
print(f"Using backend: {backend}")

num_of_shots = 1024


theta = np.pi / 4

qc = QuantumCircuit(1)
qc.rx(theta, 0)
qc.measure_all()

transpiled_circuit = transpile(qc, backend)

queued_jobs = backend.jobs(status='QUEUED')  # Note: 'QUEUED' is the correct status
print(f"queued_jobsqueue: {queued_jobs}")

queue_info = backend.queue_status()
print(f"Estimated queue time: {queue_info.estimated_queue_time} seconds")
print(f"Position in queue: {queue_info.position_in_queue}")

# Alternatively, check pending jobs
pending_jobs = backend.jobs(status='QUEUED')
if pending_jobs:
    for job in pending_jobs:
        print(f"Job ID: {job.job_id()}, Estimated completion time: {job.queue_info().estimated_complete_time}")
else:
    print("No pending jobs.")