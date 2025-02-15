from sklearn.svm import SVC 

from sklearn.metrics import accuracy_score 

from classes.my_joblib import MyJoblib

from classes.my_dill import MyDill


from classes.parameters import MyParameters
class MyModel:

    myJoblib = MyJoblib()

    myDill = MyDill()

    def trainModel(self, getQKernel, input_tr, output_tr):

        # print('myKernel.getQKernel:', myKernel.getQKernel)

        # print('1- myKernel.np:', myKernel.np)

        # print('myKernel.getQKernel: ', myKernel.getQKernel)
        svm = SVC(kernel = getQKernel).fit(input_tr, output_tr)

        print('svc:', SVC)

        MyModel.saveModel(SVC, f'saved_models/{MyParameters.modelName}')  

        # MyJoblib.saveModelToFile(SVC, f'saved_models/{MyParameters.modelName}')  

        return svm

    def predictOneItem(self, svmModel, input_test, index):

        # myJoblib = MyJoblib()

        # svm = myJoblib.l


        svmPrediction = svmModel.predict([input_test[index]])

        # print('svmPrediction: ', svmPrediction)

        return svmPrediction
    
    def predictAll(self, svmModel, input_test):

        # Check the type of the loaded object
        print(f"Type of loaded object: {type(svmModel)}")

        # If it's a class, you need to instantiate it and train it
        if isinstance(svmModel, type):
            print("The loaded object is a class, not an instance.")
        else:
            print("The loaded object is an instance.")


        svmPrediction = svmModel.predict(input_test)

        # print('svmPrediction: ', svmPrediction)

        return svmPrediction
    
    def getAccuracyScore(self, svmPredictions, output_test):

        myAccuracyScore = accuracy_score(svmPredictions, output_test)

        print('accuracy score:', myAccuracyScore)

        return myAccuracyScore
    
    def saveModel(model, fileName):
        
        # MyModel.myJoblib.saveModelToFile(model, fileName)
        MyModel.myDill.saveModelToFile(model, fileName)

    def getModel(self, fileName):
    
        # returnedModel =  MyModel.myJoblib.loadModelFromFile(fileName)
        returnedModel = MyModel.myDill.loadModelFromFile(fileName)

        return returnedModel
