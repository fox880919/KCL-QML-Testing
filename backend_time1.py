import pennylane as qml
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

backend = service.backend("ibm_brisbane")
print(f"Using backend: {backend}")

num_of_shots = 1024
# PennyLane device with shots as integer (correct)
# dev = qml.device("qiskit.remote", backend=backend, wires=1, shots=num_of_shots)

# @qml.qnode(dev)
# def circuit(theta):
#     qml.RX(theta, wires=0)
#     return qml.expval(qml.PauliZ(0))

theta = np.pi / 4
# result = circuit(theta)
# print(f"Expectation value: {result}")

# Build equivalent Qiskit circuit (no shots parameter here)
qc = QuantumCircuit(1)
qc.rx(theta, 0)
qc.measure_all()

transpiled_circuit = transpile(qc, backend)

# print(f'transpiled_circuit: {transpiled_circuit}')
properties = backend.properties()
config = backend.configuration()
status = backend.status()

print(f'properties: {properties}')
print(f'config: {config}')
print(f'status: {status}')

execution_time_per_shot = 0
for gate in transpiled_circuit.data:
    gate_name = gate.operation.name
    gate_times = [g.gate_time for g in properties.gates if g.name == gate_name]

    print(f'gate: {gate}')
    print(f'gate_name: {gate_name}')
    print(f'gate_times: {gate_times}')

    execution_time_per_shot += gate_times[0] if gate_times else 0
    print(f'execution_time_per_shot: {execution_time_per_shot}')

readout_time = getattr(config, "readout_speed", 0)
execution_time_per_shot += readout_time

# shots = dev.shots  # should be integer 1024
# print(f'shots: {shots}')
# print(f'shots.total: {shots.total}')

# print(f'shots.nshots: {shots.nshots}')
# print(f'shots._total: {shots._total}')
# num_shots = int(shots)
# print(f'num_shots: {num_shots}')

# total_time = execution_time_per_shot * shots

total_time = execution_time_per_shot * num_of_shots
queue_length = status.pending_jobs

print(f"Estimated execution time: {total_time:.20f} seconds")
print(f"Queue length: {queue_length}")