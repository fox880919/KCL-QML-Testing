import pennylane as qml
from pennylane import numpy as np

# 1. Define number of qubits (should match the feature vector size for AngleEmbedding)
num_wires = 7  # IBM Brisbane has 127 qubits, but we'll use a smaller number for this example.
             # Adjust as needed for your specific feature vector size.
             # For AngleEmbedding, num_wires should equal the number of features.
dev = None

# --- IBMQ Configuration ---
# Replace 'YOUR_API_TOKEN' with your actual IBM Quantum Experience API token.
# You might also need to specify hub, group, and project if you are not using the default open provider.
# Example: hub='ibm-q', group='open', project='main'
ibmqx_token = "YOUR_API_TOKEN" # IMPORTANT: Replace with your token
backend_name = "ibm_brisbane"   # Target IBM backend

print(f"Attempting to initialize PennyLane with IBMQ backend: {backend_name}")
print("Please ensure 'pennylane-qiskit' is installed (pip install pennylane-qiskit).")
print("You will need to replace 'YOUR_API_TOKEN' in this script with your actual IBM Quantum API token.")

try:
    # Initialize the PennyLane-Qiskit device for IBM Quantum backend
    dev = qml.device(
        "qiskit.ibmq",
        wires=num_wires,
        backend=backend_name,
        ibmqx_token=ibmqx_token
        # Optional: specify hub, group, project if needed for your account
        # hub="ibm-q",
        # group="open",
        # project="main"
    )
    print(f"Successfully initialized PennyLane device for IBMQ backend: {backend_name}")

except Exception as e:
    print(f"\n--- ERROR INITIALIZING IBMQ DEVICE ({backend_name}) ---")
    print(f"Error message: {e}")
    print("This could be due to several reasons:")
    print("  1. 'pennylane-qiskit' is not installed (run: pip install pennylane-qiskit).")
    print("  2. The IBMQ API token ('YOUR_API_TOKEN') is incorrect or not provided.")
    print(f"  3. The backend '{backend_name}' is not available to your account, is misspelled, or is offline.")
    print("  4. Network connectivity issues or IBM Quantum platform issues.")
    print("Please verify your setup and credentials.")
    print("\nFalling back to 'default.qubit' simulator for demonstration purposes.")
    dev = qml.device("default.qubit", wires=num_wires)
    print(f"Initialized fallback PennyLane device: default.qubit with {num_wires} wires.")


# 2. Define the Feature Map
# We'll use AngleEmbedding as a common example of a feature map.
# It encodes N features into the rotation angles of N qubits.
# Other feature maps like ZZFeatureMap or custom ones can also be implemented.
def feature_map_circuit(features):
    """
    Quantum circuit to implement a feature map.
    Args:
        features (np.ndarray): A 1D array of classical features to embed.
                               The length must be equal to num_wires.
    """
    if len(features) != num_wires:
        raise ValueError(f"Number of features ({len(features)}) must match number of wires ({num_wires}).")
    
    # Example: Angle Embedding feature map
    qml.AngleEmbedding(features=features, wires=range(num_wires), rotation='X')
    
    # You can add entangling layers to make the feature map more complex
    # For example, a layer of CNOTs:
    # for i in range(num_wires - 1):
    #     qml.CNOT(wires=[i, i + 1])
    
    # Another layer of rotations based on features (optional)
    # qml.AngleEmbedding(features=features, wires=range(num_wires), rotation='Y')


# 3. Create a QNode
# A QNode is a quantum function that encapsulates a quantum circuit and can be run on a quantum device.
# This QNode will apply the feature map and then, for demonstration, return the quantum state vector.
# In practical applications, you would typically return expectation values of observables.
@qml.qnode(dev)
def quantum_feature_map_qnode(features):
    """
    QNode that applies the feature map and returns the quantum state.
    Args:
        features (np.ndarray): Classical features to embed.
    Returns:
        np.ndarray: The quantum state vector (if supported by the backend and not too large).
                    For hardware backends, qml.state() might not be directly supported or efficient.
                    Consider returning qml.probs(wires=range(num_wires)) or expectation values.
    """
    feature_map_circuit(features)
    # Returning the full state vector. For real hardware or large number of qubits,
    # this can be resource-intensive or unsupported. Consider qml.probs() or qml.expval() instead.
    return qml.state()

# Example of a QNode returning probabilities (often more practical for hardware)
@qml.qnode(dev)
def quantum_feature_map_probs_qnode(features):
    feature_map_circuit(features)
    return qml.probs(wires=range(num_wires))


# 4. Example Usage
if __name__ == "__main__":
    print("\n--- EXAMPLE USAGE ---")
    # Sample classical features. The number of features must match num_wires.
    # For num_wires = 7, we need 7 features.
    sample_features = np.random.rand(num_wires) # Generates num_wires random features between 0 and 1
    print(f"Using {num_wires} wires/qubits.")
    print(f"Sample features to embed: {sample_features}")

    try:
        print(f"\nExecuting QNode 'quantum_feature_map_qnode' (returns state vector)..._token")
        # Execute the QNode that returns the state vector
        # Note: qml.state() might be slow or unsupported on real hardware for many qubits.
        if dev.name == 'default.qubit' or num_wires <= 5: # Only attempt full state for simulator or few qubits
            quantum_state = quantum_feature_map_qnode(sample_features)
            print(f"Output quantum state (first 8 elements if large):")
            if quantum_state.size > 8:
                print(quantum_state[:8])
                print("...")
            else:
                print(quantum_state)
        else:
            print("Skipping full state vector calculation for non-default backend with >5 qubits due to potential resource intensity.")
            print("Consider using the QNode that returns probabilities or expectation values.")

        # Example with probabilities (more suitable for hardware)
        print(f"\nExecuting QNode 'quantum_feature_map_probs_qnode' (returns probabilities)..._token")
        probabilities = quantum_feature_map_probs_qnode(sample_features)
        print(f"Output probabilities (first 8 elements if large):")
        if probabilities.size > 8:
            print(probabilities[:8])
            print("...")
        else:
            print(probabilities)

    except Exception as e:
        print(f"\n--- ERROR DURING QNODE EXECUTION ---")
        print(f"Error message: {e}")
        if "backend" in str(e).lower() or "token" in str(e).lower() or "authentication" in str(e).lower():
            print("This might be due to an issue with the IBMQ backend configuration, token, or availability.")
            print(f"Please ensure the backend '{backend_name}' is correctly specified, your token is valid, and the service is operational.")
        elif "Device is offline" in str(e) or "timeout" in str(e).lower():
            print(f"The backend '{backend_name}' might be offline, under maintenance, or the request timed out.")
        else:
            print("An unexpected error occurred during QNode execution.")

    print("\n--- SCRIPT FINISHED ---")
    print("To use with the actual IBM Brisbane backend:")
    print("  1. Ensure 'pennylane' and 'pennylane-qiskit' are installed.")
    print("  2. Replace 'YOUR_API_TOKEN' in the script with your valid IBM Quantum API token.")
    print(f"  3. Verify that '{backend_name}' is an available backend for your IBM Quantum account and is online.")
    print("  4. You might need to adjust 'hub', 'group', and 'project' parameters in qml.device if not using defaults.")
    print("The script includes a fallback to 'default.qubit' if IBMQ initialization fails.")

