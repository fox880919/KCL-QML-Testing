
from classes.parameters import MyParameters

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager
 


class MyFeatureMap:

    nqubits = 4
    # amplitudeNQubits = 5
    # phasenqubits = 5
    amplitudeNQubits = MyParameters.amplitudeNQubits
    phasenqubits = MyParameters.phasenqubits
    np = {}
    x_tr = []
    x_test = []
    xs_tr = []
    xs_test = []
    y_tr = [],
    y_test = []
    selectedFeatureMap = {}

    circuitDepth = 2
    nqubits = 4
    modifiedCircuit = {}

   


    def pickFeatureMapType(self, np, type = 0, components = 8):
        
        if(type == 0):
            
            # MyFeatureMap.nqubits = 4
            MyFeatureMap.nqubits = 5

            MyFeatureMap.selectedFeatureMap = MyFeatureMap.__getAmplitudeEmdedding

        elif(type == 1):

            print(f'in type == 1')
            MyFeatureMap.nqubits = components

            MyFeatureMap.nqubits = 8
            MyFeatureMap.selectedFeatureMap = MyFeatureMap.__getAngleEmdedding


        elif(type == 2):

            MyFeatureMap.np = np

            MyFeatureMap.nqubits = 4

            MyFeatureMap.selectedFeatureMap = MyFeatureMap.__getCustomEmdedding


        return MyFeatureMap.selectedFeatureMap

    # def run_IBM_Backend():

        

    def __getAmplitudeEmdedding(a, b, amplitude_qubits, total_qubits):
        # Ensure a and b are normalized
        norm_a = sum(abs(x)**2 for x in a)**0.5
        norm_b = sum(abs(x)**2 for x in b)**0.5
        a = [x / norm_a for x in a]
        b = [x / norm_b for x in b]
        
        # Create a quantum circuit
        qc = QuantumCircuit(total_qubits)
        
        # Apply amplitude embedding for `a` on amplitude_qubits
        qc.initialize(a, range(amplitude_qubits))
        
        # Apply amplitude embedding for `b` on all qubits in reverse
        qc.initialize(b, range(total_qubits))
        
        # Measure probabilities
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        statevector = job.result().get_statevector()
        
        # Return probabilities
        probabilities = Statevector(statevector).probabilities()
        return probabilities



    def __getAngleEmdedding(a, b):

        print(f'in __getAngleEmdedding')

   
    def __getCustomEmdedding(a, b):

             print(f'in __getCustomEmdedding')


    def ZZFeatureMap(nqubits, data):
        
     print(f'in ZZFeatureMap')
