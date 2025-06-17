import pennylane as qml

import pennylane_qiskit

from data.wine_data import WineData as MyData

from sklearn.decomposition import PCA

from sklearn.svm import SVC 

from sklearn.metrics import accuracy_score 

from itertools import combinations

from classes.parameters import MyParameters

from classes.time import MyTimeHelper

class MyFeatureMap:

    nqubits = 4
    amplitudeNQubits = MyParameters.amplitudeNQubits
    phasenqubits = MyParameters.phasenqubits
    np = {}

    selectedFeatureMap = {}

    circuitDepth = 2
    nqubits = 4

    def pickFeatureMapType(self, np, type = 0, components = 8):
        
        if(type == 0):
            
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

    @qml.qnode(MyParameters.getDevice())
    def __getAmplitudeEmdedding(a, b):

        qml.AmplitudeEmbedding(
        features=a, wires=range(MyFeatureMap.amplitudeNQubits), normalize=True)

        qml.AmplitudeEmbedding(
        features= b, wires=range(MyFeatureMap.nqubits), normalize=True, inverse=True)

        return qml.probs(wires = range(5))

    @qml.qnode(MyParameters.getDevice())    
    def __getAngleEmdedding(a, b):

        qml.AngleEmbedding(a, wires=range(5)) 
        qml.adjoint(qml.AngleEmbedding(b, wires=range(5))) 
        return qml.probs(wires = range(5))

    @qml.qnode(MyParameters.getDevice())    
    def __getCustomEmdedding(a, b):

        MyFeatureMap.ZZFeatureMap(MyFeatureMap.nqubits, a)
        qml.adjoint(MyFeatureMap.ZZFeatureMap)(5, b) 
        return qml.probs(wires = range(5))  

    def ZZFeatureMap(nqubits, data):
        
        nload = min(len(data), nqubits)
        for i in range(nload): qml.Hadamard(i)
        qml.RZ(2.0 * data[i], wires = i)
        for pair in list(combinations(range(nload), 2)):
            q0 = pair[0]
            q1 = pair[1]
            qml.CZ(wires = [q0, q1])
            qml.RZ(2.0 * (MyFeatureMap.np.pi - data[q0]) *
                (MyFeatureMap.np.pi - data[q1]), wires = q1)
            qml.CZ(wires = [q0, q1])
    
    def getDevice():

        dev = qml.device(MyParameters.getDevice(), wires = MyFeatureMap.amplitudeNQubits)

        if MyParameters.useIBMBackEndService == True:

            dev = qml.device(MyParameters.getDevice(), wires=MyFeatureMap.amplitudeNQubits, backend="brisbane", shots=1024)
            
        return dev
