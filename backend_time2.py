import pennylane as qml
import numpy as np
import pennylane_qiskit  # pip install pennylane-qiskit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import transpile

# 1. Define number of qubits and features
n_qubits = 2
feature_dim = 2

# 2. Create a PennyLane device (using default.qubit here just for circuit construction)
dev = qml.device("default.qubit", wires=n_qubits)

# 3. Define the feature map circuit as a PennyLane QNode
@qml.qnode(dev)
def feature_map_circuit(x):
    qml.AngleEmbedding(x, wires=range(n_qubits))
    qml.BasicEntanglerLayers(1, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Example input data
x = np.array([0.1, 0.2])

# Run circuit once (optional, to build the tape)
feature_map_circuit(x)

# 4. Extract PennyLane quantum tape
tape = feature_map_circuit.qtape

# 5. Convert PennyLane tape to Qiskit QuantumCircuit
qiskit_circuit = to_qiskit(tape)

print("Original Qiskit circuit:")
print(qiskit_circuit.draw())

# 6. Initialize IBM Quantum backend via Qiskit Runtime Service
service = QiskitRuntimeService()

backend = service.backend("ibm_brisbane")  # or your preferred backend

# 7. Transpile the circuit for the selected backend
transpiled_circuit = transpile(qiskit_circuit, backend)

print("\nTranspiled circuit:")
print(transpiled_circuit.draw())