import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager


service = QiskitRuntimeService()

backend = service.backend("ibm_brisbane")

# Step 1: Authenticate with IBM Quantum (commented out as you have it)
# ibm_token = "a2a37b0ae3d1e044e4a232ad054b7c52ee5613be17d0a48b0006b15e8e2090d02e3b1453cd07dc437529f352082c166b4eab60be49e5e5e777cab887f5aa8d36"
#   # Replace with your IBM Quantum token
# qiskit.IBMQ.save_account(ibm_token)  # Save your IBM Quantum token
# qiskit.IBMQ.load_account()  # Load your IBM Quantum account

# Step 2: Define the quantum device using the IBM backend
n_qubits = 4  # Adjust based on your dataset or requirements
# --- FIX: Added c_wires for classical bit mapping ---
dev = qml.device("qiskit.remote", wires=n_qubits, backend=backend, c_wires=range(n_qubits))


# Step 3: Define the quantum kernel using the amplitude embedding feature map
@qml.qnode(dev)
def quantum_kernel(x1, x2):
    # Amplitude embedding for the first data point
    qml.AmplitudeEmbedding(features=x1, wires=range(n_qubits), normalize=True, pad_with=0)
    
    # Amplitude embedding for the second data point (inverse)
    qml.adjoint(qml.AmplitudeEmbedding(features=x2, wires=range(n_qubits), pad_with=0, normalize=True))  
    
      
    # Measure the overlap between the two states
    return qml.probs(wires=range(n_qubits))

# Step 4: Compute the kernel matrix
def compute_kernel_matrix(X1, X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    kernel_matrix = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            # Check for data point length matching n_qubits
            # Pad or truncate if necessary. Here, we'll assume X is prepared to match 2^n_qubits for AmplitudeEmbedding
            # If your data points are shorter than 2^n_qubits, AmplitudeEmbedding will pad with zeros if normalize=True and pad_with=0 (which you have)
            # If your data points are longer, it will raise an error unless truncated.
            
            # The [0] is correct if you're taking the probability of the |0...0> state after the overlap measurement
            kernel_matrix[i, j] = quantum_kernel(X1[i], X2[j])[0]  
    
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

# IMPORTANT: AmplitudeEmbedding expects the number of features to be 2^n_qubits.
# Your current n_qubits is 4, meaning it expects 2^4 = 16 features.
# The wine dataset has 13 features. You need to handle this mismatch.
# Options:
# 1. Increase n_qubits to accommodate all features (e.g., if 13 features, you'd need 4 qubits for 2^4=16, which you have).
#    If you had 17 features, you'd need 5 qubits (2^5=32).
# 2. Reduce the number of features used from your dataset to match 2^n_qubits.
# 3. Use a different embedding that doesn't have this 2^n_qubits requirement, or allows for more flexible padding.
# Since you have 13 features and n_qubits=4, AmplitudeEmbedding will implicitly pad the remaining 16-13=3 dimensions with zeros, which is acceptable.

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