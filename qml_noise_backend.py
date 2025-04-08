from qiskit import IBMQ
#IBMQ.save_account('YOUR_API_TOKEN')  # One-time setup

from qiskit import IBMQ, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from qiskit.providers.ibmq import least_busy

import pennylane as qml
from pennylane import numpy as np

# 1. Load IBMQ and get real backend noise
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 2 and
                                       not b.configuration().simulator and b.status().operational==True))
print(f"Using backend: {backend.name()}")

noise_model = NoiseModel.from_backend(backend)
coupling_map = backend.configuration().coupling_map
basis_gates = noise_model.basis_gates

# 2. Set up PennyLane device with noise model via Qiskit
dev = qml.device('qiskit.aer', wires=2, noise_model=noise_model, basis_gates=basis_gates, coupling_map=coupling_map)

# Optional: Reduce shots for speed
# dev = qml.device('qiskit.aer', wires=2, shots=1024, noise_model=noise_model, basis_gates=basis_gates)

# 3. Define quantum circuits as before
def feature_map(x):
    for i in range(2):
        qml.Hadamard(wires=i)
        qml.RZ(x[i], wires=i)
        qml.RY(x[i], wires=i)
    qml.CZ(wires=[0, 1])

def variational_circuit(params):
    for i in range(2):
        qml.RY(params[i], wires=i)
        qml.RZ(params[i+2], wires=i)
    qml.CNOT(wires=[0, 1])

@qml.qnode(dev)
def circuit(x, params):
    feature_map(x)
    variational_circuit(params)
    return qml.expval(qml.PauliZ(0))

# You can continue training and testing like before