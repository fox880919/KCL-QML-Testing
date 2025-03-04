import sys


sys.path.insert(0, './data')

from qsvm.my_model import MyModel

from data.data_manager import DataManager

from classes.parameters import MyParameters

from data.my_dataframe import MyDataFrame

from qsvm.my_model import MyModel

myDataFrame = MyDataFrame()

myModel = MyModel()

dataManager =DataManager()

myModel = MyModel()

def getElementsScores():
    

    nfold = 16

    totalNotEqual = 0

    originalAccuracyScore = 0
    mrsAccuracyScores = 0
    total = 0

    dataType = MyParameters.dataType

    np, train_data_list, test_data_list = dataManager.getListOfFoldDatabyNumber(dataType)

    for fold_index in range(len(train_data_list)):

        x_tr, y_tr = train_data_list[fold_index]
        x_test, y_test = test_data_list[fold_index]


        originalModelDigit = 0
        originalModelFeatureMapDigit = 0


        OriginalodelName = 'SVM'+ str(0) + str(originalModelDigit)+ '-' + str(originalModelFeatureMapDigit) + '-' + str(fold_index) + '-of-' + str(nfold)

        fullOriginalModelName = f'saved_models/{OriginalodelName}'
        # print('fullModelName: ', fullModelName)                    

        OriginalSavedModel = myModel.getModel(fullOriginalModelName)

        OriginalTrainedModel = OriginalSavedModel().fit(x_tr, y_tr)

        # print('OriginalTrainedModel: ', OriginalTrainedModel)
        # originalAccuracyScore = myDataFrame.getModelScoreValue(0, 0, fold_index, nfold)

        # if fold_index == 0:

        # # print('len(x_test): ', len(x_test))
        # # for x_test_index in len(x_test): 

        # for x_test_index in range(len(x_test)):

        
        #     resultedLabel = myModel.predictOneItem(OriginalTrainedModel, x_test, x_test_index)
        
        #     print('x_test_index: ', x_test_index)
        #     print('result: ', resultedLabel)

        # # for one_test_data in x_test:


        # print('for fold_index', fold_index)

        # print('len(x_tr)', len(x_tr))
        # print('len(x_test)', len(x_test))

        # print('x_tr[0]', x_tr[0])

        # originalAccuracyScore = myDataFrame.getModelScoreValue(0, 0, fold_index, nfold)

        # print('OriginalTrainedModel: ', OriginalTrainedModel) 
        # print('originalAccuracyScore: ', originalAccuracyScore)



        for mrNumber in range(1, 6):

            if mrNumber == 1:
                    
                # for mrValue in range(2, 20):
                for mrValue in range(2, 4):
                    
                    modelName = 'SVM'+ str(0) + str(mrNumber)+ '-' + str(mrValue) + '-' + str(fold_index) + '-of-' + str(nfold)

                    fullModelName = f'saved_models/{modelName}'
                    # print('fullModelName: ', fullModelName)                    

                    SavedModel = myModel.getModel(fullModelName)

                    trainedModel = SavedModel().fit(x_tr, y_tr)

                    for x_test_index in range(len(x_test)):

        
                        originalResultedLabel = myModel.predictOneItem(OriginalTrainedModel, x_test, x_test_index)
                    
                        resultedLabel = myModel.predictOneItem(trainedModel, x_test, x_test_index)

                        if not originalResultedLabel == resultedLabel: 

                            total = total + 1


                        else:
                            print('they are equal')
                            # print(f'for x_test[{x_test_index}], originalResultedLabel = {originalResultedLabel} and resultedLabel = {resultedLabel}')
                            
                            # print('originalResultedLabel == resultedLabel', originalResultedLabel == resultedLabel)

                        # print('x_test_index: ', x_test_index)
                        # print('result: ', resultedLabel)


                    # for one_test in x_test:



                    # mrsAccuracyScores = mrsAccuracyScores + myModel.getModel(mrNumber, mrValue, fold_index, nfold)

    print('total: ', total)  



getElementsScores()


