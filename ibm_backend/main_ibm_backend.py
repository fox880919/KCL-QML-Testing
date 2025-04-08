import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import qiskit  # Import the qiskit module

# Step 1: Authenticate with IBM Quantum
ibm_token = "a2a37b0ae3d1e044e4a232ad054b7c52ee5613be17d0a48b0006b15e8e2090d02e3b1453cd07dc437529f352082c166b4eab60be49e5e5e777cab887f5aa8d36"
  # Replace with your IBM Quantum token
qiskit.IBMQ.save_account(ibm_token)  # Save your IBM Quantum token
qiskit.IBMQ.load_account()  # Load your IBM Quantum account

# Step 2: Define the quantum device using the IBM backend
n_qubits = 4  # Adjust based on your dataset or requirements
dev = qml.device("qiskit.ibmq", wires=n_qubits, backend="ibmq_qasm_simulator")  # Use the desired backend

# Step 3: Define the quantum kernel using the amplitude embedding feature map
@qml.qnode(dev)
def quantum_kernel(x1, x2):
    # Amplitude embedding for the first data point
    qml.AmplitudeEmbedding(features=x1, wires=range(n_qubits), normalize=True)
    
    # Amplitude embedding for the second data point (inverse)
    qml.AmplitudeEmbedding(features=x2, wires=range(n_qubits), normalize=True, inverse=True)
    
    # Measure the overlap between the two states
    return qml.probs(wires=range(n_qubits))

# Step 4: Compute the kernel matrix
def compute_kernel_matrix(X1, X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    kernel_matrix = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            kernel_matrix[i, j] = quantum_kernel(X1[i], X2[j])[0]  # Probability of the first state
    
    return kernel_matrix

# Step 5: Load and preprocess the wine dataset
data = load_wine()
X = data.data
y = data.target

# Use only two classes for binary classification
X = X[y != 2]
y = y[y != 2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Compute the kernel matrices for training and testing
K_train = compute_kernel_matrix(X_train, X_train)
K_test = compute_kernel_matrix(X_test, X_train)

# Step 7: Train the SVM model using the quantum kernel
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

# Step 8: Predict on the test set
y_pred = svm.predict(K_test)

# Step 9: Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")