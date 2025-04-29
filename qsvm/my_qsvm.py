# import pennylane as qml

# from data.wine_data import WineData as MyData

# from sklearn.decomposition import PCA

# from sklearn.svm import SVC 

# from sklearn.metrics import accuracy_score 

from my_feature_map import MyFeatureMap

# from my_feature_map_qiskit import MyQiskitFeatureMap

from my_kernel import MyKernel

from my_model import MyModel

from classes.parameters import MyParameters

class MyQSVM:

    np = {}

    mySelectedFeatureMap = {}

    myModel = MyModel()

    myKernel = MyKernel()


    def startSVC(self, np, input_tr, input_test, output_tr, output_test, modelName, fold_index, featureMapType = 0, components = 8):
        
        MyQSVM.np = np

        # fullModelName = 'saved_models/' + modelName
        fullModelName = modelName

        myFeatureMap = MyFeatureMap()

        MyQSVM.myKernel.np = MyQSVM.np

        if MyParameters.useQiskit == False:

            MyQSVM.mySelectedFeatureMap = myFeatureMap.pickFeatureMapType(np, featureMapType, components)

        else:

            # MyQSVM.mySelectedFeatureMap = MyQiskitFeatureMap.pickFeatureMapType(np, featureMapType, components)
            MyQSVM.mySelectedFeatureMap = myFeatureMap.pickFeatureMapType(np, featureMapType, components)


        MyQSVM.myKernel.mySelectedFeatureMap = MyQSVM.mySelectedFeatureMap

        # svm = MyQSVM.myModel.trainModel(MyQSVM.getQKernel, input_tr, output_tr)

        if not MyParameters.useTrainedModel:

            # print('train model started')

            svm = MyQSVM.myModel.trainModel(MyQSVM.myKernel.getQKernel, input_tr, output_tr, fullModelName, fold_index)

            # print('train model ended ')

            # MyQSVM.myModel.saveModel(svm, 'saved_models/svm00')
        else:

            print('get saved model')
            savedSVC = MyQSVM.myModel.getModel(fullModelName)
            svm = savedSVC().fit(input_tr, output_tr)

        # return 
        svmPredictions = MyQSVM.myModel.predictAll(svm, input_test)

        myAccuracyScore = MyQSVM.myModel.getAccuracyScore(svmPredictions, output_test)

        # print('accuracy score:', myAccuracyScore)

        return myAccuracyScore
    
    def __getQKernel(A, B):

        return MyQSVM.np.array([[MyQSVM.mySelectedFeatureMap(a, b)[0] for b in B] for a in A])


