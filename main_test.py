
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Wine dataset


print('get wine data')

data = load_wine()
X = data.data
y = data.target

print('got wine data')

# For simplicity, let's consider a binary classification problem
# We will classify class 0 vs class 1
# X = X[y != 2]
# y = y[y != 2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f'wine data, len(X_train): {len(X_train)}')

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('wine data transformed')

# Apply PCA with n_components = 8

# print('before pca, X_train[0]: ', X_train[0])
# pca = PCA(n_components=8)

# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

# print('after pca, X_train[0]: ', X_train[0])


# Define the number of qubits (now equal to the number of PCA components)
# n_qubits = X_train.shape[1]

n_qubits= 4
print(f'n_qubits:{n_qubits}')

# Define the device
dev = qml.device("default.qubit", wires=n_qubits)

inputNumber = 0

# Define the quantum circuit for angle embedding
# @qml.qnode(dev)
# def amplitude_embedding(x1, x2):

#     global inputNumber
#     inputNumber = inputNumber + 1
#     print(f'amplitude embedding inputNumber: {inputNumber}')
#     # Angle embedding for the first data point
#     for i in range(n_qubits):
#         qml.RX(x1[i], wires=i)
    
#     # Angle embedding for the second data point
#     for i in range(n_qubits):
#         qml.RX(x2[i], wires=i)
    
#     # Measure the overlap between the two states
#     return qml.expval(qml.Hermitian(np.eye(2**n_qubits), wires=range(n_qubits)))
#     #  qml.AmplitudeEmbedding(
#     #     x1, wires=range(n_qubits), pad_with=0, normalize=True)

@qml.qnode(dev)
def amplitude_embedding(a, b):
        

        global inputNumber
        inputNumber = inputNumber + 1
        print(f'amplitude embedding inputNumber: {inputNumber}')

        qml.AmplitudeEmbedding(
        a, wires=range(n_qubits), pad_with=0, normalize=True)


        qml.adjoint(qml.AmplitudeEmbedding(
        b, wires=range(n_qubits), pad_with=0, normalize=True))


        return qml.probs(wires = range(n_qubits))


# Define the kernel matrix function
def kernel_matrix(X1, X2):

    print(f'kernel_matrix, len(X1): {len(X1)}')
    print(f'kernel_matrix, len(X1[0]): {len(X1[0])}')

    return np.array([[amplitude_embedding(x1, x2) for x2 in X2] for x1 in X1])
    # return np.array([[amplitude_embedding(a, b)[0] for b in X2] for a in X1])



print('start kernel')



# Compute the kernel matrix for the training data
K_train = kernel_matrix(X_train, X_train)

print('start SVC training')

# Train the SVM using the quantum kernel
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

print('Compute the kernel matrix for the test data')

# Compute the kernel matrix for the test data
K_test = kernel_matrix(X_test, X_train)

print('Predict the labels for the test data')

# Predict the labels for the test data
y_pred = svm.predict(K_test)


print('Calculate the accuracy')

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")