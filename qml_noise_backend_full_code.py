# =============================
# INSTALL (if not already done)
# pip install pennylane qiskit qiskit-aer scikit-learn matplotlib
# =============================

import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq import least_busy

# ------------------------------------
# 1. LOAD IBM BACKEND AND NOISE MODEL
# ------------------------------------
IBMQ.load_account()  # Load stored IBMQ token
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 2
                                       and not b.configuration().simulator
                                       and b.status().operational==True))
print(f"Using backend: {backend.name()}")

noise_model = NoiseModel.from_backend(backend)
coupling_map = backend.configuration().coupling_map
basis_gates = noise_model.basis_gates

# -------------------------------
# 2. QUANTUM DEVICE WITH NOISE
# -------------------------------
dev = qml.device("qiskit.aer", wires=2, shots=1024,
                 noise_model=noise_model,
                 basis_gates=basis_gates,
                 coupling_map=coupling_map)

# -------------------------------
# 3. DATASET PREPROCESSING
# -------------------------------
data = load_wine()
X = data.data
y = data.target

# Binary classification: class 0 vs 1
X = X[y != 2]
y = y[y != 2]

# Scale and reduce to 2 features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = PCA(n_components=2).fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# -------------------------------
# 4. QUANTUM CIRCUITS
# -------------------------------
def feature_map(x):
    for i in range(2):
        qml.Hadamard(wires=i)
        qml.RZ(x[i], wires=i)
        qml.RY(x[i], wires=i)
    qml.CZ(wires=[0, 1])

def variational_circuit(params):
    for i in range(2):
        qml.RY(params[i], wires=i)
        qml.RZ(params[i+2], wires=i)
    qml.CNOT(wires=[0, 1])

@qml.qnode(dev)
def circuit(x, params):
    feature_map(x)
    variational_circuit(params)
    return qml.expval(qml.PauliZ(0))

# -------------------------------
# 5. QSVM Training Functions
# -------------------------------
def predict(x, params):
    predictions = [np.sign(circuit(xi, params)) for xi in x]
    return np.array([(1 if p >= 0 else 0) for p in predictions])

def cost(params, x, y):
    predictions = predict(x, params)
    return np.mean((predictions - y)**2)

# -------------------------------
# 6. TRAIN QSVM
# -------------------------------
np.random.seed(0)
params = np.random.randn(4, requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.1)

epochs = 20
for epoch in range(epochs):
    params, c = opt.step_and_cost(lambda p: cost(p, X_train, y_train), params)
    print(f"Epoch {epoch + 1:02d}: cost = {c:.4f}")

# -------------------------------
# 7. EVALUATE QSVM
# -------------------------------
y_pred = predict(X_test, params)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy (with noise): {acc:.2f}")