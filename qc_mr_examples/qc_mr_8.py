import matplotlib.pyplot as plt 

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
 
# Create a simple circuit
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0,1)
circuit.cx(0,2)
circuit.measure_all()

# circuit.draw('mpl') 
# plt.show()

# Define backend with 3 qubits
backend = GenericBackendV2(num_qubits=3)
 
# Transpile and run
transpiled_circuit = transpile(circuit, backend)

transpiled_circuit.draw(output="mpl")
plt.show()
