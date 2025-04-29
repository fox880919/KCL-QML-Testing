import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from pennylane.templates import AmplitudeEmbedding

# Load the complete Wine dataset with all 3 classes
wine = load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Normalize data points to be valid quantum states (unit norm)
X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

# Number of qubits is determined by the number of features (must be power of 2)
n_features = X_train.shape[1]
n_qubits = int(np.ceil(np.log2(n_features)))
required_dim = 2**n_qubits

# Pad the data with zeros to make dimension a power of 2
X_train_padded = np.zeros((X_train.shape[0], required_dim))
X_train_padded[:, :n_features] = X_train
X_test_padded = np.zeros((X_test.shape[0], required_dim))
X_test_padded[:, :n_features] = X_test

print(f"Original features: {n_features}")
print(f"Using {n_qubits} qubits for {required_dim}-dimensional embedding")

# Quantum circuit for the quantum kernel
dev = qml.device("default.qubit", wires=n_qubits)

inputNumber = 0

@qml.qnode(dev)
def kernel_circuit(x1, x2):

    global inputNumber
    inputNumber = inputNumber + 1
    print(f'amplitude embedding inputNumber: {inputNumber}')

    """Quantum kernel circuit using amplitude embedding."""
    # Embed first data point
    qml.AmplitudeEmbedding(x1, wires=range(n_qubits), normalize=True)
    
    # Apply inverse of second data point embedding using adjoint
    qml.adjoint(AmplitudeEmbedding)(x2, wires=range(n_qubits), normalize=True)
    
    # Return the probability of the all-zero state
    # Changed to use qml.probs() and index properly
    return qml.probs(wires=range(n_qubits))

@qml.qnode(dev)
def __getAmplitudeEmdedding(a, b):

        global inputNumber
        inputNumber = inputNumber + 1
        print(f'amplitude embedding inputNumber: {inputNumber}')

        qml.AmplitudeEmbedding(a, wires=range(n_qubits), normalize=True)

        qml.adjoint(qml.AmplitudeEmbedding(b, wires=range(n_qubits), normalize=True))

        return qml.probs(wires = range(n_qubits))

def kernel_matrix(A, B):
    """Compute the kernel matrix between two sets of samples."""
    # Need to extract the first probability (all zeros state)
    return np.array([[__getAmplitudeEmdedding(a, b)[0] for b in B] for a in A])

# Compute the kernel matrices
print("Computing training kernel matrix...")
K_train = kernel_matrix(X_train_padded, X_train_padded)

print("Computing testing kernel matrix...")
K_test = kernel_matrix(X_test_padded, X_train_padded)

# Train a multi-class SVM using one-vs-rest strategy
print("Training SVM with One-vs-Rest strategy...")
svm = OneVsRestClassifier(SVC(kernel="precomputed"))
svm.fit(K_train, y_train)

# Predict and evaluate
train_preds = svm.predict(K_train)
test_preds = svm.predict(K_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print("\nClassification Results:")
print(f"Training accuracy: {train_acc:.4f}")
print(f"Testing accuracy: {test_acc:.4f}")

# Print per-class metrics
from sklearn.metrics import classification_report
print("\nDetailed Classification Report:")
print(classification_report(y_test, test_preds, target_names=wine.target_names))