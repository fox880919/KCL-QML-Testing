import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Import necessary Qiskit components for IBM backend and circuit building
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator # For local simulation to get statevectors
import os # For securely loading API token from environment variables

# --- 1. Data Loading and Preprocessing ---

# Load the wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# For simplicity and amplitude embedding, let's pick two features.
# Amplitude embedding typically works best when the number of features
# is a power of 2, as it maps directly to qubit amplitudes.
# We'll use the first two features (alcohol and malic acid) which fits 1 qubit.
X_selected = X[:, [0, 1]] # Alcohol and Malic acid

# We will classify between two classes for simplicity (e.g., class 0 vs. class 1).
# Filter data to only include samples from classes 0 and 1.
X_binary = X_selected[(y == 0) | (y == 1)]
y_binary = y[(y == 0) | (y == 1)]

# Scale the data to be between 0 and 1. This is crucial for amplitude embedding
# as the amplitudes of a quantum state must be normalized.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_binary)

# Split the dataset into training and testing sets to evaluate model performance.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42
)

print(f"Original number of features: {X.shape[1]}")
print(f"Selected number of features: {X_selected.shape[1]}")
print(f"Number of samples for binary classification: {X_scaled.shape[0]}")
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of testing samples: {X_test.shape[0]}")
print("\n--- Data Preprocessing Complete ---\n")

# --- 2. Quantum Circuit Definition (Feature Map in Qiskit) ---

# Determine the number of qubits required for amplitude embedding.
# For N features, you need ceil(log2(N)) qubits.
# With 2 features, ceil(log2(2)) = 1 qubit is needed.
num_qubits = 1

def qiskit_amplitude_embedding_circuit(features, num_qubits):
    """
    Constructs a Qiskit QuantumCircuit for amplitude embedding.
    This circuit takes a feature vector and encodes it into the amplitudes
    of the quantum state. Qiskit's initialize method handles normalization.

    Args:
        features (np.ndarray): The input classical feature vector.
        num_qubits (int): The number of qubits for the circuit.

    Returns:
        qiskit.QuantumCircuit: The quantum circuit with amplitude embedding.
    """
    circuit = QuantumCircuit(num_qubits)
    # The 'initialize' method takes a state vector (amplitudes) and prepares
    # the quantum state. It automatically normalizes if the input is not.
    # For a single qubit and 2 features [f0, f1], the state vector is [f0, f1].
    # Qiskit's `initialize` requires the state to be a list of complex numbers.
    # We convert numpy array to list.
    circuit.initialize(features.tolist(), range(num_qubits))
    return circuit

# --- Configure IBM Quantum Backend (Optional - for running circuits later, not for kernel calc here) ---
# IMPORTANT: Never hardcode your API token directly in the script for security reasons.
# This section is kept for demonstrating how to set up the PennyLane Qiskit device,
# though for the kernel calculation below, we will use AerSimulator directly.

IBMQ_TOKEN = os.getenv("IBM_QUANTUM_API_TOKEN", "YOUR_IBM_QUANTUM_API_TOKEN")

try:
    service = QiskitRuntimeService(channel="ibm_quantum", token=IBMQ_TOKEN)
    # This `dev` object is primarily for PennyLane QNodes if you had any
    # that directly return expectation values. For the kernel calculation,
    # we're using Qiskit's AerSimulator directly for state vector access.
    backend_name = "ibmq_qasm_simulator" # Or choose a real device like "ibm_osaka"
    backend_instance = service.backend(backend_name)
    print(f"Connecting to IBM Quantum backend: {backend_name}")
    # Initialize a PennyLane device for potential future QNodes that might run on IBM backend
    dev_qiskit_remote = qml.device("qiskit.remote", wires=num_qubits, backend=backend_instance, shots=1024)
    print("PennyLane device configured for IBM Quantum (dev_qiskit_remote).")

except Exception as e:
    print(f"Could not load IBM Quantum account or connect to specified backend. Error: {e}")
    print("Proceeding without direct IBM Quantum backend connection for PennyLane device.")
    dev_qiskit_remote = None # Indicate that the IBM Qiskit remote device is not available

print("--- Qiskit Feature Map Function Defined ---\n")

# --- 3. Quantum Kernel Function (Manual Fidelity Calculation using Qiskit) ---

# Initialize a local AerSimulator to get state vectors for fidelity calculation
# This simulator is efficient for getting state vectors.
statevector_simulator = AerSimulator(method='statevector')

def quantum_kernel_manual_qiskit(x_i, x_j):
    """
    Computes the fidelity kernel value between two data points x_i and x_j
    using Qiskit for circuit execution and state vector manipulation.
    The fidelity kernel is defined as |<phi(x_i)|phi(x_j)>|^2.

    Args:
        x_i (np.ndarray): The first data point.
        x_j (np.ndarray): The second data point.

    Returns:
        float: The fidelity kernel value.
    """
    # Create quantum circuits for x_i and x_j using Qiskit's amplitude embedding
    qc_i = qiskit_amplitude_embedding_circuit(x_i, num_qubits)
    qc_j = qiskit_amplitude_embedding_circuit(x_j, num_qubits)

    # Simulate the circuits to get their state vectors
    # Transpile for the simulator (optional for statevector method, but good practice)
    tqc_i = transpile(qc_i, statevector_simulator)
    tqc_j = transpile(qc_j, statevector_simulator)

    # Run the circuits to get the results. The 'result().get_statevector()'
    # method is specific to statevector simulations.
    job_i = statevector_simulator.run(tqc_i)
    result_i = job_i.result()
    state_i = Statevector(result_i.get_statevector(qc_i)) # Get Statevector object

    job_j = statevector_simulator.run(tqc_j)
    result_j = job_j.result()
    state_j = Statevector(result_j.get_statevector(qc_j)) # Get Statevector object

    # Compute the inner product between the two quantum states: <state_i | state_j>.
    # The 'inner' method of Statevector calculates <self | other>.
    inner_product = state_i.inner(state_j)

    # The fidelity is the squared absolute value of the inner product.
    fidelity = np.abs(inner_product)**2
    return fidelity

def compute_gram_matrix_qiskit(X_A, X_B=None):
    """
    Computes the Gram matrix (or kernel matrix) for a given dataset(s)
    using the Qiskit-based manual fidelity calculation.

    Args:
        X_A (np.ndarray): The first set of data points (e.g., training data).
        X_B (np.ndarray, optional): The second set of data points (e.g., test data).
                                     If None, the Gram matrix is computed between X_A and itself.

    Returns:
        np.ndarray: The computed Gram matrix.
    """
    if X_B is None:
        X_B = X_A

    gram_matrix = np.zeros((X_A.shape[0], X_B.shape[0]))

    for i, x_i in enumerate(X_A):
        for j, x_j in enumerate(X_B):
            # Use the Qiskit-based kernel calculation
            gram_matrix[i, j] = quantum_kernel_manual_qiskit(x_i, x_j)

            if (i * X_B.shape[0] + j + 1) % 50 == 0 or \
               (i == X_A.shape[0] - 1 and j == X_B.shape[0] - 1):
                print(f"Computed {i * X_B.shape[0] + j + 1} / {X_A.shape[0] * X_B.shape[0]} kernel values...")

    return gram_matrix

print("--- Quantum Kernel Function Defined (Manual Fidelity using Qiskit) ---\n")

# --- 4. Training and Evaluation ---

print("Calculating training Gram matrix (this might take a moment)...")
# Compute the training Gram matrix using the Qiskit-based function
K_train = compute_gram_matrix_qiskit(X_train)
print("Training Gram matrix calculated.\n")

print("Calculating testing Gram matrix (this might take a moment)...")
# Compute the testing Gram matrix using the Qiskit-based function
K_test = compute_gram_matrix_qiskit(X_test, X_train)
print("Testing Gram matrix calculated.\n")

# Initialize and train the Support Vector Classifier (SVC).
qsvm = SVC(kernel='precomputed')
qsvm.fit(K_train, y_train)

# Make predictions on the test set
y_pred = qsvm.predict(K_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"QSVM Accuracy on the test set: {accuracy * 100:.2f}%")
print("\n--- QSVM Training and Evaluation Complete ---")
