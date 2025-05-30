from sklearn.svm import SVC 

from sklearn.metrics import accuracy_score 

from classes.my_joblib import MyJoblib

from classes.my_dill import MyDill

from classes.parameters import MyParameters

class MyModel:

    myJoblib = MyJoblib()

    myDill = MyDill()

    def trainModel(self, getQKernel, input_tr, output_tr, modelName, fold_index):

        # print('myKernel.getQKernel:', myKernel.getQKernel)

        # print('1- myKernel.np:', myKernel.np)

        # print('myKernel.getQKernel: ', myKernel.getQKernel)

        print('training model started')

        if MyParameters.showProgressDetails:
            # svm = SVC(kernel = getQKernel, verbose= True).fit(input_tr, output_tr)
            svm = SVC(kernel = getQKernel).fit(input_tr, output_tr)


        else:
            svm = SVC(kernel = getQKernel).fit(input_tr, output_tr)

        print('training model ended')

        # print('svc:', SVC)

        # don't use in testing

        print('saving model started')

        savingModelName = MyParameters.getSavingModelFolderName()
        #models naming 1
        # MyModel.saveModel(SVC, f'{MyParameters.savedModelsFolder}/{modelName}')  

        #models naming 2
        MyModel.saveModel(SVC, f'{savingModelName}/{modelName}')  

        print('saving model ended')

        # MyModel.saveModel(SVC, f'saved_models/{modelName}')  

        # MyJoblib.saveModelToFile(SVC, f'saved_models/{MyParameters.modelName}')  

        return svm

    def predictOneItem(self, svmModel, input_test, index, mrValue = 1):

        # print('svmModel: ', svmModel)
        # print('input_test: ', input_test)
        # print('index: ', index)

        # myJoblib = MyJoblib()

        # svm = myJoblib.l


        single_input = mrValue * input_test[index]

        svmPrediction = svmModel.predict([single_input])
        # svmPrediction = svmModel.predict(single_input)
        

        # print('svmPrediction: ', svmPrediction)

        return svmPrediction
    
    def predictAll(self, svmModel, input_test):

        # Check the type of the loaded object
        # print(f"Type of loaded object: {type(svmModel)}")

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
    

    
