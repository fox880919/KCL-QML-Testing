from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.algorithms.optimizers import COBYLA
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import combinations
import numpy as np


class MyQiskitFeatureMap:

    def __init__(self, nqubits=4, circuit_depth=2, components=8):
        self.nqubits = nqubits
        self.circuit_depth = circuit_depth
        self.components = components
        self.x_tr = []
        self.x_test = []
        self.y_tr = []
        self.y_test = []
        self.selected_feature_map = None

    def pick_feature_map_type(self, type=0):
        if type == 0:
            # Amplitude Embedding
            self.selected_feature_map = self.__get_amplitude_embedding
        elif type == 1:
            # Angle Embedding
            self.selected_feature_map = self.__get_angle_embedding
        elif type == 2:
            # Custom Embedding
            self.selected_feature_map = self.__get_custom_embedding

    def __implement_pca(self, components=8):
        pca = PCA(n_components=components)
        self.x_tr = pca.fit_transform(self.x_tr)
        self.x_test = pca.transform(self.x_test)

    def __get_amplitude_embedding(self, data):
        qc = QuantumCircuit(self.nqubits)
        data_norm = np.linalg.norm(data)
        if data_norm > 0:
            data = data / data_norm
        qc.initialize(data, range(self.nqubits))
        return qc

    def __get_angle_embedding(self, data):
        qc = QuantumCircuit(self.nqubits)
        for i, value in enumerate(data):
            qc.rx(value, i)
        return qc

    def __get_custom_embedding(self, data):
        qc = QuantumCircuit(self.nqubits)
        zz_feature_map = ZZFeatureMap(feature_dimension=self.nqubits, reps=self.circuit_depth)
        param_values = {p: v for p, v in zip(zz_feature_map.parameters, data)}
        zz_feature_map = zz_feature_map.assign_parameters(param_values)
        qc.compose(zz_feature_map, inplace=True)
        return qc

    def start_svc(self, x_tr, x_test, y_tr, y_test):
        # Feature maps for kernel calculations
        def kernel(a, b):
            qasm_sim = Aer.get_backend('qasm_simulator')
            quantum_instance = QuantumInstance(backend=qasm_sim, shots=1024)
            kernel_value = []
            for x in a:
                row = []
                for y in b:
                    # Prepare circuits for the kernel
                    circuit = self.selected_feature_map(x).compose(self.selected_feature_map(y).inverse())
                    circuit = transpile(circuit, backend=qasm_sim)
                    result = quantum_instance.execute(circuit)
                    counts = result.get_counts()
                    prob_0 = counts.get('0' * self.nqubits, 0) / 1024
                    row.append(prob_0)
                kernel_value.append(row)
            return np.array(kernel_value)

        svc = SVC(kernel=kernel).fit(x_tr, y_tr)
        y_pred = svc.predict(x_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))