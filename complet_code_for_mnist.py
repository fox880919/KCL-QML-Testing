import pennylane as qml
from pennylane import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Step 1: Load Fashion MNIST
print("📦 Loading Fashion MNIST...")
fashion_mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
X = fashion_mnist.data.astype('float32') / 255.0
y = fashion_mnist.target.astype('int')

# Optional: use a smaller subset for speed (recommended)
sample_size = 2000
X = X[:sample_size]
y = y[:sample_size]

# Step 2: Reduce dimensionality (QSVM doesn’t like 784 dims)
print("📉 Applying PCA...")
pca_components = 4  # Try 2, 4, or 6
pca = PCA(n_components=pca_components)
X_reduced = pca.fit_transform(X)

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Setup Quantum Kernel
n_qubits = pca_components
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CZ(wires=[i, i+1])

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(n_qubits))

def quantum_kernel(X1, X2):
    print("🧠 Computing quantum kernel matrix...")
    return np.array([[kernel_circuit(x1, x2)[0] for x2 in X2] for x1 in X1])

# Step 6: Compute Kernel Matrices
X_train_np, X_test_np = np.array(X_train), np.array(X_test)
K_train = quantum_kernel(X_train_np, X_train_np)
K_test = quantum_kernel(X_test_np, X_train_np)

# Step 7: Train QSVM (SVM with precomputed quantum kernel)
print("🎯 Training QSVM (SVC with quantum kernel)...")
clf = SVC(kernel='precomputed', decision_function_shape='ovr')
clf.fit(K_train, y_train)

# Step 8: Predict and Evaluate
y_pred = clf.predict(K_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ QSVM Accuracy (multiclass): {acc:.4f}")

# Optional: Plot confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - QSVM on Fashion MNIST")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()