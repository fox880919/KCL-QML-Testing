import pennylane as qml
import numpy as np

from qiskit import transpile

from qiskit_ibm_runtime import QiskitRuntimeService


service = QiskitRuntimeService()

backend = service.backend(
            "ibm_brisbane"
        )

print(f'backend: {backend}')
# Define a quantum device
# dev = qml.device("default.qubit", wires=1)
dev = qml.device("qiskit.remote", backend=backend, wires=1, shots = 1024)

# Define a quantum node (quantum function)
@qml.qnode(dev)
def circuit(theta):
    qml.RX(theta, wires=0)  # Apply an RX rotation
    return qml.expval(qml.PauliZ(0))  # Measure expectation value of PauliZ on wire 0


theta = 0.5

qiskit_circuit = dev.compile(circuit, max_iterations=1)(theta)

# Transpile for the selected backend
transpiled_circuit = transpile(qiskit_circuit, backend)

# Estimate execution time per shot
properties = backend.properties()
config = backend.configuration()
status = backend.status()


# Execute the quantum node
theta = np.pi / 4
result = circuit(theta)
print(f"Expectation value: {result}")