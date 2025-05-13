import pennylane as qml
from qiskit_ibm_runtime import QiskitRuntimeService

# Load IBM backend
service = QiskitRuntimeService()
print("Available backends:", [backend.name for backend in service.backends()])

backend = service.backend("ibm_brisbane")     

print(f"backend:{backend}")
