import numpy as np
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import QSVM
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report

# Load the Wine dataset
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

# For simplicity, let's focus on binary classification (class 0 vs class 1)
# Filter to only include classes 0 and 1
mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Normalize the data to be between -1 and 1 (important for quantum feature maps)
normalizer = MinMaxScaler((-1, 1))
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

# Reduce dimensionality to 2 features for better visualization and performance
X_train = X_train[:, :2]
X_test = X_test[:, :2]

# Set up the quantum feature map and quantum instance
feature_dim = X_train.shape[1]
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement="linear")

# Use statevector simulator for highest accuracy
quantum_instance = QuantumInstance(
    Aer.get_backend("statevector_simulator"),
    shots=1024,
    seed_simulator=42,
    seed_transpiler=42,
)

# Create quantum kernel
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

# Create QSVM model
qsvm = QSVM(quantum_kernel, X_train, y_train)

# Train the model (this may take some time)
qsvm.fit(X_train, y_train)

# Evaluate the model
train_score = qsvm.score(X_train, y_train)
test_score = qsvm.score(X_test, y_test)

print(f"Training accuracy: {train_score:.2f}")
print(f"Testing accuracy: {test_score:.2f}")

# Make predictions
y_pred = qsvm.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Visualize the decision boundary
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model, title):
    h = 0.02  # step size in the mesh
    cmap = ListedColormap(["#FFAAAA", "#AAFFAA"])
    
    # Create mesh to plot in
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for each point in mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(X_train, y_train, qsvm, "QSVM Decision Boundary (Training Data)")