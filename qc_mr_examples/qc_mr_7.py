import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit, transpile
# from qiskit.providers.fake_provider import FakeBackend 
from qiskit.providers.fake_provider import GenericBackendV2 

# Create a sample quantum circuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.z(2)

# qc.draw('mpl') 
# plt.show()

#0:2, 1: 0, 2: 1
# coupling_map = [[0, 2], [1,0], [2,1]]

# coupling_map = [[0, 0], [1,1], [2,2]]

backend = GenericBackendV2(num_qubits=3) 

# backend = GenericBackendV2(num_qubits=3, coupling_map = coupling_map) 

# Define the desired qubit mapping (e.g., map qubit 0 to physical qubit 2)
# initial_layout = {0: 2, 1: 0, 0: 1} 


# Transpile the circuit with the specified initial layout
transpiled_circuit = transpile(qc, backend=backend)

transpiled_circuit.draw(output="mpl")
plt.show()
