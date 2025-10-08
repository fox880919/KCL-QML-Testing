#
# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script demonstrates how to build a Quantum Support Vector Machine (QSVM)
with PennyLane and compares its performance with and without a noise model.
It also applies a metamorphic relation (scaling inputs) to test the model's behavior.
"""

import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set a random seed for reproducibility
np.random.seed(42)

# --- 1. Data Preparation ---
# Generate a synthetic dataset
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y = y * 2 - 1  # Convert labels from {0, 1} to {-1, 1}

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale the features to be between -pi and pi, which is a good range for angle encoding
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Metamorphic Relation: Scale inputs by a factor of 11 ---
scaling_factor = 11
X_train_meta = X_train_scaled * scaling_factor
X_test_meta = X_test_scaled * scaling_factor


# --- 2. Quantum Kernel Definition ---
n_qubits = 2
# Define the quantum device for the ideal (noiseless) simulation
dev_ideal = qml.device("default.qubit", wires=n_qubits)

# Define the quantum device for the noisy simulation
# We will use the 'default.mixed' device which allows for noise modeling
dev_noisy = qml.device("default.mixed", wires=n_qubits)

def feature_map(x):
    """Encodes a 2D data point into the quantum state of 2 qubits."""
    qml.RY(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.CNOT(wires=[0, 1])

def noisy_feature_map(x, noise_param):
    """
    Feature map with depolarizing noise applied after each gate.
    This simulates the effect of noise on a real quantum device.
    """
    qml.RY(x[0], wires=0)
    qml.DepolarizingChannel(noise_param, wires=0)
    qml.RX(x[1], wires=1)
    qml.DepolarizingChannel(noise_param, wires=1)
    qml.CNOT(wires=[0, 1])
    # Apply noise to both qubits after the CNOT
    qml.DepolarizingChannel(noise_param, wires=0)
    qml.DepolarizingChannel(noise_param, wires=1)

def adjoint_noisy_feature_map(x, noise_param):
    """
    Manually defines the adjoint of the noisy_feature_map.
    The adjoint of a sequence of operations is the sequence of the adjoints
    of each operation, in reverse order. The DepolarizingChannel is self-adjoint.
    """
    # Adjoint of the CNOT and its subsequent noise
    qml.DepolarizingChannel(noise_param, wires=1)
    qml.DepolarizingChannel(noise_param, wires=0)
    qml.CNOT(wires=[0, 1])

    # Adjoint of the RX and its subsequent noise
    qml.DepolarizingChannel(noise_param, wires=1)
    qml.RX(-x[1], wires=1)

    # Adjoint of the RY and its subsequent noise
    qml.DepolarizingChannel(noise_param, wires=0)
    qml.RY(-x[0], wires=0)


def kernel_circuit(x1, x2, use_noise=False, noise_param=0.0):
    """
    Quantum circuit to compute the kernel value between two data points.
    The kernel is computed as the fidelity between the two quantum states
    prepared by the feature map.
    """
    if use_noise:
        noisy_feature_map(x1, noise_param)
        # We use the manually defined adjoint for the noisy circuit
        adjoint_noisy_feature_map(x2, noise_param)
    else:
        feature_map(x1)
        qml.adjoint(feature_map)(x2)

    # The kernel value is the probability of measuring the all-zero state
    return qml.probs(wires=range(n_qubits))

def get_kernel_function(use_noise=False, noise_param=0.0):
    """
    Returns a kernel function that can be used by scikit-learn's SVC.
    This function computes the Gram matrix.
    """
    device = dev_noisy if use_noise else dev_ideal
    
    # Create a QNode for the kernel circuit
    # The QNode is the fundamental computational unit in PennyLane
    qnode = qml.QNode(kernel_circuit, device, interface="auto", diff_method=None)

    def kernel(X1, X2):
        """Computes the Gram matrix between two datasets."""
        gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                # The [0] index gets the probability of the |00> state
                prob = qnode(x1, x2, use_noise=use_noise, noise_param=noise_param)[0]
                gram_matrix[i, j] = prob
        return gram_matrix

    return kernel

# --- 3. Training and Evaluation ---

# --- Noiseless QSVM ---
print("Training the original noiseless QSVM...")
kernel_ideal = get_kernel_function(use_noise=False)
svm_ideal = SVC(kernel=kernel_ideal)
svm_ideal.fit(X_train_scaled, y_train)

# --- Noisy QSVM ---
print("\nTraining the original noisy QSVM (40% depolarizing noise)...")
noise_level = 0.4 # 40% depolarizing noise
kernel_noisy = get_kernel_function(use_noise=True, noise_param=noise_level)
svm_noisy = SVC(kernel=kernel_noisy)
svm_noisy.fit(X_train_scaled, y_train)

# --- Metamorphic Noiseless QSVM ---
print(f"\nTraining the metamorphic noiseless QSVM (inputs scaled by {scaling_factor})...")
svm_ideal_meta = SVC(kernel=kernel_ideal)
svm_ideal_meta.fit(X_train_meta, y_train)

# --- Metamorphic Noisy QSVM ---
print(f"\nTraining the metamorphic noisy QSVM (inputs scaled by {scaling_factor})...")
svm_noisy_meta = SVC(kernel=kernel_noisy)
svm_noisy_meta.fit(X_train_meta, y_train)


# --- 4. Results Comparison ---
print("\n--- Results ---")

# Evaluate the original models
accuracy_ideal = svm_ideal.score(X_test_scaled, y_test)
accuracy_noisy = svm_noisy.score(X_test_scaled, y_test)
print(f"Accuracy of the original noiseless QSVM: {accuracy_ideal:.4f}")
print(f"Accuracy of the original noisy QSVM (40% noise): {accuracy_noisy:.4f}")

# Evaluate the metamorphic models
accuracy_ideal_meta = svm_ideal_meta.score(X_test_meta, y_test)
accuracy_noisy_meta = svm_noisy_meta.score(X_test_meta, y_test)
print(f"\nAccuracy of the metamorphic noiseless QSVM (scaled inputs): {accuracy_ideal_meta:.4f}")
print(f"Accuracy of the metamorphic noisy QSVM (scaled inputs): {accuracy_noisy_meta:.4f}")


print("\n--- Conclusion ---")
print("The results show how noise impacts performance on both original and transformed data.")
print("Comparing the original and metamorphic results reveals the model's sensitivity to input scaling, a key aspect of metamorphic testing.")
