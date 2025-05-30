import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Data preparation
def generate_data():
    from sklearn.datasets import load_wine
    data = load_wine()
    X = data.data[:, :2]  # Use the first two features for simplicity
    y = (data.target != 0).astype(int)  # Binary classification (class 0 vs others)
    y = 2 * y - 1  # Convert labels to -1 and 1
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Quantum device and kernel
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    for i in range(len(x)):
        qml.RY(x[i], wires=i)
    qml.CZ(wires=[0, 1])

def quantum_kernel(x1, x2):
    qml.templates.embeddings.AngleEmbedding(x1, wires=range(n_qubits))
    feature_map(x1)
    feature_map(x2).inv()

kernel_device = qml.kernels.QuantumKernel(
    kernel=quantum_kernel, wires=range(n_qubits)
)

def train_qsvm():
    X_train, X_test, y_train, y_test = generate_data()
    
    # Compute kernel matrices
    train_kernel_matrix = kernel_device.kernel_matrix(X_train)
    test_kernel_matrix = kernel_device.kernel_matrix(X_test, X_train)
    
    # Train classical SVM with quantum kernel
    clf = SVC(kernel="precomputed")
    clf.fit(train_kernel_matrix, y_train)
    
    accuracy = clf.score(test_kernel_matrix, y_test)
    print(f"QSVM Accuracy: {accuracy}")
    return accuracy

if __name__ == "__main__":
    train_qsvm()