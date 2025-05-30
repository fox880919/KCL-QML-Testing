from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager
 
print('backend_time4.py')
service = QiskitRuntimeService()
 
# Create a new circuit with two qubits (first argument) and two classical
# bits (second argument)
qc = QuantumCircuit(2, 2)
 
# Add a Hadamard gate to qubit 0
qc.h(0)
 
# Perform a controlled-X gate on qubit 1, controlled by qubit 0
qc.cx(0, 1)
 
qc.measure(0, 0)
qc.measure(1, 1)
 

backend = service.backend("ibm_brisbane")

print(f'backend: {backend}')

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
print(f'pm: {pm}')

isa_circuit = pm.run(qc)
# print(f'isa_circuit: {isa_circuit}')


# Create a Sampler object
sampler = Sampler(backend)
# print(f'sampler: {sampler}')

for job in service.jobs():
    # print(job.job_id(), job.status())
    print(f'job.job_id(): {job.job_id()}, and job.status(): {job.status()}')
    if job.status().name == "QUEUED":
        job.cancel()
        print(f"Cancelled job {job.job_id()}.")

# service.jobs(job_id="d0kgk3ccrrag008ncnrg").cancel()
# service.jobs(job_id="d0kfeptvpqf000810rr0").cancel()
# service.jobs(job_id="d0kfeptvpqf000810rr0").cancel()

# Submit the circuit to the sampler
job = sampler.run([isa_circuit])

# print(f'job: {job}')

print(f'job.usage_estimation: {job.usage_estimation}')