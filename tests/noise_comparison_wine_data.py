import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2 # A 5-qubit fake backend
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA # Import PCA for dimensionality reduction

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Data Loading and Preprocessing with PCA ---

# Load Wine dataset and select two classes for binary classification
wine = load_wine()
X, y = wine.data, wine.target
class_0_idx, class_1_idx = np.where(y == 0), np.where(y == 1)
X_binary = np.vstack((X[class_0_idx], X[class_1_idx]))
y_binary = np.hstack((y[class_0_idx], y[class_1_idx]))
y_binary = np.where(y_binary == 0, -1, 1)

# Apply PCA to reduce the 13 features down to 2
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_binary)
print(f"Original feature count: {X_binary.shape[1]}")
print(f"Reduced feature count after PCA: {X_pca.shape[1]}")

# Scale the new 2-feature dataset to the range [0, π] for angle encoding
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_pca)

# Split the final, reduced-feature dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# --- 2. Quantum Circuit and Backend Setup ---

# Dynamically set the number of qubits to match the data features (now 2)
n_features = X_train.shape[1]
n_qubits = n_features
print(f"✅ Using {n_qubits} qubits to match the {n_features} data features.")

# Define the quantum feature map
def create_feature_map():
    """Create a parameterized quantum circuit for the feature map."""
    x = ParameterVector('x', length=n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(x[i], i)
    return qc, x

feature_map, x_params = create_feature_map()

# Initialize a fake backend for local simulation. FakeManilaV2 (5 qubits) is now sufficient.
backend = FakeManilaV2()
print(f"Using fake backend: {backend.name}")

# --- 3. Quantum Kernel Computation ---

def compute_kernel_matrix(X1, X2=None, shots=1024):
    """Computes the kernel matrix by running SWAP test circuits."""
    if X2 is None:
        X2 = X1
    
    n_samples1, n_samples2 = X1.shape[0], X2.shape[0]
    kernel_matrix = np.zeros((n_samples1, n_samples2))

    options = {"simulator": {"seed_simulator": 42}}
    sampler = SamplerV2(backend=backend, options=options)

    pubs, pub_indices = [], []

    for i in range(n_samples1):
        for j in range(n_samples2):
            # SWAP test now requires 1 + 2*2 = 5 qubits
            qc = QuantumCircuit(1 + 2 * n_qubits, 1)

            # Encode data points
            qc.compose(feature_map.assign_parameters(X1[i]), qubits=range(1, n_qubits + 1), inplace=True)
            qc.compose(feature_map.assign_parameters(X2[j]), qubits=range(n_qubits + 1, 1 + 2 * n_qubits), inplace=True)
            
            # Perform the SWAP test
            qc.h(0)
            for k in range(n_qubits):
                qc.cswap(0, k + 1, n_qubits + k + 1)
            qc.h(0)
            qc.measure(0, 0)
            
            pubs.append(transpile(qc, backend=backend))
            pub_indices.append((i, j))
    
    print(f"Submitting {len(pubs)} circuits ({1 + 2 * n_qubits} qubits each) to the local simulator...")
    job = sampler.run(pubs, shots=shots)
    result = job.result()

    # Process results
    for idx, pub_result in enumerate(result):
        i, j = pub_indices[idx]
        counts = pub_result.data.c.get_counts()
        p0 = counts.get('0', 0) / shots
        kernel_matrix[i, j] = 2 * p0 - 1
    
    return kernel_matrix

# --- 4. Training and Evaluation ---

shots = 1024

print("\nComputing training kernel matrix...")
K_train = compute_kernel_matrix(X_train, shots=shots)

print("Computing testing kernel matrix...")
K_test = compute_kernel_matrix(X_test, X_train, shots=shots)

print("Training SVM with the precomputed quantum kernel...")
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

train_preds = svm.predict(K_train)
test_preds = svm.predict(K_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print("\n--- Results ---")
print(f"Training accuracy: {train_acc:.4f}")
print(f"Testing accuracy: {test_acc:.4f}")