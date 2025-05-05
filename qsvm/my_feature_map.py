import pennylane as qml

from data.wine_data import WineData as MyData

from sklearn.decomposition import PCA

from sklearn.svm import SVC 

from sklearn.metrics import accuracy_score 

from itertools import combinations

from classes.parameters import MyParameters

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

    # myParameters = MyParameters()

    #noise model 1
    noise_model = qml.transforms.insert(
    qml.DepolarizingChannel,  # You can use other channels like AmplitudeDamping, PhaseDamping, etc.
    op_positions = "all",     # Apply to all operations
    op_args=lambda obj: (0.05,)  # Noise parameter (e.g., 0.05 depolarization probability)
    )

    #or noisy device 3
    # dev = qml.device("default.mixed", wires=MyFeatureMap.amplitudeNQubits)


    def pickFeatureMapType(self, np, type = 0, components = 8):
        
        if(type == 0):
            
            # MyFeatureMap.nqubits = 4
            MyFeatureMap.nqubits = 5

            MyFeatureMap.selectedFeatureMap = MyFeatureMap.__getAmplitudeEmdedding

        elif(type == 1):

            # print('getting angle embedding')
            MyFeatureMap.nqubits = components

            MyFeatureMap.nqubits = 8
            # MyFeatureMap.__implementPCA(components)
            MyFeatureMap.selectedFeatureMap = MyFeatureMap.__getAngleEmdedding


        elif(type == 2):

            MyFeatureMap.np = np

            MyFeatureMap.nqubits = 4

            MyFeatureMap.selectedFeatureMap = MyFeatureMap.__getCustomEmdedding


        # print('MyFeatureMap.selectedFeatureMap: ', MyFeatureMap.selectedFeatureMap)

        return MyFeatureMap.selectedFeatureMap
        

    # @qml.qnode(qml.device("lightning.qubit", wires = 4))
    # @noise_model
    # @qml.qnode(qml.device("lightning.qubit", wires = amplitudeNQubits))
    #for noise use default.mixed
    # @qml.qnode(qml.device("default.mixed", wires = amplitudeNQubits))
    @qml.qnode(qml.device(MyParameters.getDevice(), wires = amplitudeNQubits))
    def __getAmplitudeEmdedding(a, b):
        
        # print(f'amplitudeNQubits {MyFeatureMap.amplitudeNQubits}')

        if MyParameters.showProgressDetails:
            # print(f'amplitude embedding roundNumber: {MyParameters.roundNumber}')
            print(f'amplitude embedding inputNumber: {MyParameters.inputNumber}')

            # print(f'len(A): {len(a)}, and len(B): {len(b)}')
        
        MyParameters.inputNumber = MyParameters.inputNumber + 1           

        qml.AmplitudeEmbedding(
        a, wires=range(MyFeatureMap.amplitudeNQubits), pad_with=0, normalize=True)

        ## or manual noise 1
        if MyParameters.applyDepolarizingChannelNoise == True:
            for wire in range(MyFeatureMap.amplitudeNQubits):
                qml.DepolarizingChannel(MyParameters.depolarizingChannelNoise, wires=wire) 


        qml.adjoint(qml.AmplitudeEmbedding(
        b, wires=range(MyFeatureMap.nqubits), pad_with=0, normalize=True))

        ## or manual noise 2
        if MyParameters.applyPhaseDampingNoise == True:
            for wire in range(MyFeatureMap.amplitudeNQubits):
                qml.BitFlip(MyParameters.bitFlipNoise, wires=wire) 

        ## or manual noise 3
        if MyParameters.applyBitFlipNoise == True:
            for wire in range(MyFeatureMap.amplitudeNQubits):
                qml.BitFlip(MyParameters.bitFlipNoise, wires=wire) 

        return qml.probs(wires = range(MyFeatureMap.nqubits))


    def __implementPCA(components = 8):
        
        pca = PCA(n_components = components)
        MyFeatureMap.xs_tr = pca.fit_transform(MyFeatureMap.x_tr)
        MyFeatureMap.xs_test = pca.transform(MyFeatureMap.x_test)

    # @qml.qnode(qml.device("lightning.qubit", wires = MyParameters.pca_components))
    @qml.qnode(qml.device(MyParameters.getDevice(), wires = MyParameters.pca_components))
    def __getAngleEmdedding(a, b):

        qml.AngleEmbedding(a, wires=range(MyFeatureMap.nqubits)) 
        qml.adjoint(qml.AngleEmbedding(b, wires=range(MyFeatureMap.nqubits))) 
        return qml.probs(wires = range(MyFeatureMap.nqubits))

    # @qml.qnode(dev)
    # @qml.qnode(qml.device("lightning.qubit", wires = 4))
    @qml.qnode(qml.device(MyParameters.getDevice(), wires = 4))
    def __getCustomEmdedding(a, b):

        MyFeatureMap.ZZFeatureMap(MyFeatureMap.nqubits, a)
        qml.adjoint(MyFeatureMap.ZZFeatureMap)(MyFeatureMap.nqubits, b) 
        return qml.probs(wires = range(MyFeatureMap.nqubits))  

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
    
    # def startSVC(self, input_tr, input_test, output_tr, output_test):

    #     svm = SVC(kernel = MyFeatureMap.__getQKernel).fit(input_tr, output_tr)

    #     print('accuracy score:')
    #     print(accuracy_score(svm.predict(input_test), output_test))

    # def __getQKernel(A, B):
    #     return MyFeatureMap.np.array([[MyFeatureMap.kernel_circ(a, b)[0] for b in B] for a in A])


    # def startSVCWithModifiedCircuit(self, input_tr, input_test, output_tr, output_test, circuitDepth, nqubits, modifiedCircuit):

    #     MyQSVM.circuitDepth = circuitDepth
    #     MyQSVM.nqubits = nqubits
    #     MyQSVM.modifiedCircuit = modifiedCircuit

    #     svm = SVC(kernel = MyQSVM.__getQKernelWithModifiedCircuit).fit(input_tr, output_tr)

    #     print('accuracy score:')
    #     print(accuracy_score(svm.predict(input_test), output_test))
    
    # def __getQKernelWithModifiedCircuit(A, B):

        
    #     return MyQSVM.np.array([[MyQSVM.modifiedCircuit(a, b)[0] for b in B] for a in A])


    # custom noise 4
    # def custom_noise(op, **kwargs):
    # # Apply the original operation
    # op()
    
    # # Add custom noise
    # for wire in op.wires:
    #     qml.DepolarizingChannel(0.03, wires=wire)
    #     qml.PhaseDamping(0.02, wires=wire)