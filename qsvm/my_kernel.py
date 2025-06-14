import pennylane as qml

from pennylane import numpy as np

from data.wine_data import WineData as MyData

from sklearn.decomposition import PCA

from sklearn.svm import SVC 

from sklearn.metrics import accuracy_score 

from my_feature_map import MyFeatureMap

from classes.parameters import MyParameters

from my_model import MyModel

class MyKernel:

    np = {}

    mySelectedFeatureMap = {}

    myModel = MyModel()

    # def startSVC(self, np, input_tr, input_test, output_tr, output_test, featureMapType = 0, components = 8):
        
    #     MyKernel.np = np

    #     # print('in start SVC')

    #     # print('np is: ', np)

    #     # print('np is: ', MyKernel.np)

    #     myFeatureMap = MyFeatureMap()
        
    #     MyKernel.mySelectedFeatureMap = myFeatureMap.pickFeatureMapType(np, featureMapType, components)

    #     # svm = SVC(kernel = MyKernel.__getQKernel).fit(input_tr, output_tr)
        
    #     # svmPrediction = svm.predict(input_test)

    #     # myAccuracyScore = accuracy_score(svmPrediction, output_test)

    #     # # myAccuracyScore = accuracy_score(svm.predict(input_test), output_test)

    #     svm = MyKernel.myModel.trainModel(MyKernel.__getQKernel, input_tr, output_tr)

    #     svmPredictions = MyKernel.myModel.predictAll(svm, input_test)

    #     myAccuracyScore = MyKernel.myModel.getAccuracyScore(svmPredictions, output_test)

    #     # print('accuracy score:', myAccuracyScore)

    #     return myAccuracyScore


    def getQKernel(self, A, B):

        print(f'getQKernel, (number of samples) len(A): {len(A)}')

        # print(f'getQKernel, (number of demensions) len(A[0]): {len(A[0])}')

        # print(f"Evaluating kernel between: {A} and {B}")
        MyParameters.inputNumber = 1
        
        if MyParameters.showProgressDetails:
            print(f'QKernel roundNumber: {MyParameters.roundNumber}')
            # print(f'len(A): {len(A)}, and len(B): {len(B)}')
        # print('2- MyKernel.np: ', self.np)

        if MyParameters.usePrecomputedKernel == False:
            return self.np.array([[self.mySelectedFeatureMap(a, b)[0] for b in B] for a in A])
        
        else: 

            n1 = A.shape[0]
            n2 = B.shape[0]
            kernel_matrix = np.zeros((n1, n2))
            
            for i in range(n1):
                for j in range(n2):
                    kernel_matrix[i, j] = self.mySelectedFeatureMap(A[i], B[j])[0]  # Probability of the first state
            
            return kernel_matrix


    def __getQKernel(A, B):

        return MyKernel.np.array([[MyKernel.mySelectedFeatureMap(a, b)[0] for b in B] for a in A])


    # def startSVCWithModifiedCircuit(self, input_tr, input_test, output_tr, output_test, circuitDepth, nqubits, modifiedCircuit):

    #     MyKernel.circuitDepth = circuitDepth
    #     MyKernel.nqubits = nqubits
    #     MyKernel.modifiedCircuit = modifiedCircuit

    #     svm = SVC(kernel = MyKernel.__getQKernelWithModifiedCircuit).fit(input_tr, output_tr)

    #     print('accuracy score:')
    #     print(accuracy_score(svm.predict(input_test), output_test))
    

    # def __getQKernelWithModifiedCircuit(A, B):

    #     return MyKernel.np.array([[MyKernel.modifiedCircuit(a, b)[0] for b in B] for a in A])



