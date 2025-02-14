import matplotlib.pyplot as plt 

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
 
# Create a simple circuit
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0,1)
circuit.cx(0,2)
# circuit.measure_all()

circuit.draw('mpl') 
plt.show()
 

# result = backend.run(transpiled_circuit).result()

from qiskit_ibm_runtime import QiskitRuntimeService
ibmToken = "a2a37b0ae3d1e044e4a232ad054b7c52ee5613be17d0a48b0006b15e8e2090d02e3b1453cd07dc437529f352082c166b4eab60be49e5e5e777cab887f5aa8d36"
service = QiskitRuntimeService(channel="ibm_quantum", token=ibmToken)

# backends = service.backends()

# for backend in backends:
#     print(backend)

# # <IBMBackend('ibm_brisbane')>
# # <IBMBackend('ibm_kyiv')>
# # <IBMBackend('ibm_sherbrooke')>

# backend = service.backends(name="ibm_brisbane")[0] 
# # backend = service.backends(name="ibm_kyiv")[0] 
# # backend = service.backends(name="ibm_sherbrooke")[0] 

# # backend = GenericBackendV2(num_qubits=3)


# transpiled_circuit = transpile(circuit, backend)

# # print(transpiled_circuit)
# # transpiled_circuit.draw(output="mpl")
# transpiled_circuit.draw(output="mpl", scale= 0.1)

# # fig, ax = plt.subplots(figsize=(20, 10))  # Adjust figure size as needed
# # transpiled_circuit.draw('mpl', ax=ax) 

# plt.show()
# # plt.show(block=True)
