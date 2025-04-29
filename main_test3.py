
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from pennylane.templates import AmplitudeEmbedding

# import sys
# from data.data_manager import DataManager
from datetime import datetime
# from qsvm.my_qsvm import MyQSVM

# myQSVM = MyQSVM()

# dataManager = DataManager()

# np, x_tr, x_test, y_tr, y_test = dataManager.getDatabyNumber(0)


wine = load_wine()
x = wine.data
y = wine.target

# For binary classification between class 0 and 1
# mask = (y == 0) | (y == 1)
# X = X[mask]
# y = y[mask]

x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)
x_test = scaler.transform(x_test)

# Normalize data points to be valid quantum states (unit norm)
# X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
# X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f'starttime: {time}')

# Number of qubits should match the number of features
n_qubits = x_tr.shape[1]
print(f"Number of qubits: {n_qubits}")

# required_dim = 2**n_qubits
# Pad the data with zeros to make dimension a power of 2
# X_train_padded = np.zeros((X_train.shape[0], required_dim))
# X_train_padded[:, :n_features] = X_train

# X_test_padded = np.zeros((X_test.shape[0], required_dim))
# X_test_padded[:, :n_features] = X_test

# for fold_index in range(len(train_data_list)):


# Quantum circuit for the quantum kernel
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    """Quantum kernel circuit using amplitude embedding."""
    # Embed first data point
    AmplitudeEmbedding(x1, wires=range(n_qubits), normalize=True)
    # Adjoint embedding of second data point
    AmplitudeEmbedding(x2, wires=range(n_qubits), normalize=True, inverse=True)
    # Return the probability of the all-zero state
    return qml.probs(wires=range(n_qubits))[0]


inputNumber = 0

@qml.qnode(dev)
def __getAmplitudeEmdedding(a, b):

        global inputNumber
        inputNumber = inputNumber + 1
        print(f'amplitude embedding inputNumber: {inputNumber}')

        qml.AmplitudeEmbedding(a, wires=range(n_qubits), pad_with=0, normalize=True)

        qml.adjoint(qml.AmplitudeEmbedding(b, wires=range(n_qubits), pad_with=0, normalize=True))

        return qml.probs(wires = range(n_qubits))


def kernel_matrix(A, B):
    """Compute the kernel matrix between two sets of samples."""
    # return np.array([[kernel_circuit(a, b) for b in B] for a in A])
    return np.array([[__getAmplitudeEmdedding(a, b) for b in B] for a in A])



# print(f'starttime of kfold({fold_index}): {time}')

# Compute the kernel matrices
print("Computing training kernel matrix...")
k_train = kernel_matrix(x_tr, x_tr)


print("Computing testing kernel matrix...")
k_test = kernel_matrix(x_test, x_tr)

print("Training SVM...")
svm = SVC(kernel="precomputed")
svm.fit(k_train, k_test)

test_preds = svm.predict(k_test)

test_acc = accuracy_score(y_test, test_preds)

print(f"Testing accuracy: {test_acc:.4f}")

# x_tr, x_test, y_tr, y_test = dataModification(x_tr, x_test, y_tr, y_test)
