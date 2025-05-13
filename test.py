from qiskit_ibm_runtime import QiskitRuntimeService, Options
import pennylane as qml
from pennylane import numpy as np

# 2. Load service and connect to Brisbane
service = QiskitRuntimeService()


try:
    # Get Brisbane backend with explicit hub/group/project
    backend = service.backend(
        "ibm_brisbane"
    )
    print(f"Successfully connected to {backend.name}")
except Exception as e:
    print(f"Error accessing Brisbane: {e}")
    print("Available backends:", [b.name for b in service.backends()])
    exit()

# 3. Configure PennyLane device
# dev = qml.device(
#     "qiskit.ibm",
#     # backend='ibm_brisbane',
#     backend='qiskit.remote',

#     wires=127,  # Brisbane has 127 qubits
#     shots=4000,  # Recommended for noise mitigation
#     initial_layout=list(range(127)),  # Map logical to physical qubits
#     optimization_level=3  # Aggressive transpilation
# )

# dev = qml.device(
#     "qiskit.ibmq",
#     wires=2,
#     backend='ibmq_brisbane',
# )

print(f'backend: {backend}')

dev = qml.device('qiskit.remote', wires=5, backend=backend)

print(f'dev: {dev}')

# print(f'print(dev.operations): {print(dev.operations)}')

# print(f'dev.capabilities()["operations"]: {dev.capabilities()["operations"]}')
# print(f'print(dev.operations): {print(dev.operations)}')

# 4. Define circuit optimized for Brisbane's topology
# @qml.qnode(dev)
# def brisbane_circuit():
#     # Example circuit using Brisbane's heavy-hexagon connectivity
#     qml.Hadamard(wires=0)
#     qml.CNOT(wires=[0, 1])
#     qml.CNOT(wires=[1, 2])  # Follows Brisbane's connectivity
#     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

# # 5. Run with error mitigation
# options = Options(
#     resilience_level=1,  # Basic error mitigation
#     execution={"shots": 4000}
# )


@qml.qnode(dev)
def feature_map_circuit(a):
    # Normalize data and embed it using Amplitude Embedding
    # qml.templates.AmplitudeEmbedding(data, wires=range(3), normalize=True)
    # qml.CNOT(wires=[0, 1])
    # qml.CNOT(wires=[1, 2])
    # return qml.probs(wires=range(3))
    qml.AmplitudeEmbedding(
    a, wires=range(5), pad_with=0, normalize=True)

    # qml.adjoint(qml.AmplitudeEmbedding(
    #     b, wires=range(5), pad_with=0, normalize=True))  
    return qml.probs(wires = range(5))





# Step 4: Input classical data
# Example data of size 2^3 = 8 to fit 3 qubits
data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
data = data / np.linalg.norm(data)  # Normalize manually (optional)

# Step 5: Execute the circuit
probs = feature_map_circuit(data)
print(f"Output probabilities: {probs}")

# print("Running on Brisbane...")
# result = brisbane_circuit()
# print("Result:", result)