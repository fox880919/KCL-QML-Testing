import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Generate a toy dataset
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y = 2 * y - 1  # Convert labels to -1 and 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a quantum device
n_qubits = X.shape[1]
dev = qml.device("default.qubit", wires=n_qubits)

# Define a feature map
def feature_map(x):
    for i in range(len(x)):
        qml.RX(x[i], wires=i)
    for i in range(len(x) - 1):
        qml.CZ(wires=[i, i + 1])

# Quantum kernel
def quantum_kernel(x1, x2):
    qml.templates.embeddings.AngleEmbedding(x1, wires=range(n_qubits))
    feature_map(x1)
    feature_map(x2).inv()

kernel_device = qml.kernels.QuantumKernel(
    kernel=quantum_kernel, wires=range(n_qubits), shots=None
)