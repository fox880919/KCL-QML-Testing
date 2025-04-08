import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the wine dataset
data = load_wine()
X = data.data
y = data.target

# Make it binary (e.g., class 0 vs class 1)
X = X[y != 2]
y = y[y != 2]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reduce to 2D for visualization & to fit on 2 qubits
pca = PCA(n_components=2)
X = pca.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PennyLane format
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Step 2: Define quantum circuit
n_qubits = 2

dev = qml.device("default.qubit", wires=n_qubits)
#noise
# dev = qml.device("default.mixed", wires=n_qubits)

# Feature map
def feature_map(x):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(x[i], wires=i)
        qml.RY(x[i], wires=i)
        #noise 1
        # qml.DepolarizingChannel(0.01, wires=i)  # Add depolarizing noise

    qml.CZ(wires=[0, 1])
    #noise 2
    # qml.DepolarizingChannel(0.01, wires=1)  # Noise after entanglement


# Variational circuit
def variational_circuit(params):
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
        qml.RZ(params[i + n_qubits], wires=i)
        #noise 3
        # qml.BitFlip(0.01, wires=i)  # Add bit-flip noise


    qml.CNOT(wires=[0, 1])
     #noise 4
    # qml.PhaseDamping(0.01, wires=1)  # Add phase damping


@qml.qnode(dev)
def circuit(x, params):
    feature_map(x)
    variational_circuit(params)
    return qml.expval(qml.PauliZ(0))

# Step 3: Prediction function
def predict(x, params):
    predictions = [np.sign(circuit(xi, params)) for xi in x]
    return np.array([(1 if p >= 0 else 0) for p in predictions])