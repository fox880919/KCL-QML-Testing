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

def getElementsComparison():

    nfold = 16

    totalNotEqual = 0

    originalAccuracyScore = 0
    mrsAccuracyScores = 0
    total = 0

    dataType = MyParameters.dataType

    np, train_data_list, test_data_list = dataManager.getListOfFoldDatabyNumber(dataType)

    # print('len(train_data_list): ', len(train_data_list))

    #length is 16
    for fold_index in range(len(train_data_list)):

        #1
        print('for fold_index: ', fold_index)

        x_tr, y_tr = train_data_list[fold_index]
        x_test, y_test = test_data_list[fold_index]

        # print(f'for fold_index: {fold_index}, len(x_test): {len(x_test)}' )
        # continue
        originalModelDigit = 0
        originalModelFeatureMapDigit = 0


        # OriginalModelName = 'SVM'+ str(0) + str(originalModelDigit)+ '-' + str(originalModelFeatureMapDigit) + '-' + str(fold_index) + '-of-' + str(nfold)
        OriginalModelName = MyParameters.getModelName(originalModelDigit, originalModelFeatureMapDigit, fold_index, nfold)
        
        # fullOriginalModelName = f'saved_models/{OriginalModelName}'
        fullOriginalModelName = MyParameters.getFullPathModelName(OriginalModelName)
        
        # print('fullModelName: ', fullModelName)                    

        OriginalSavedModel = myModel.getModel(fullOriginalModelName)

        OriginalTrainedModel = OriginalSavedModel().fit(x_tr, y_tr)

        for mrNumber in range(1, 6):
            
            #length is 1
            if mrNumber == 1:

                #2 
                # print('for mrNumber: ', mrNumber)

                #length is 18
                for mrValue in range(2, 20):
                # for mrValue in range(2, 4):

                    #3 
                    # print('for mrValue: ', mrValue)

                    modelName = 'SVM'+ str(0) + str(mrNumber)+ '-' + str(mrValue) + '-' + str(fold_index) + '-of-' + str(nfold)

                    fullModelName = f'saved_models/{modelName}'

                    print(f'fullModelName: ${fullModelName}')

                    SavedModel = myModel.getModel(fullModelName)

                    trainedModel = SavedModel().fit(x_tr, y_tr)

                    #length is 11 or 12
                    for x_test_index in range(len(x_test)):
                        
                        #4
                        # print('for x_test_index: ', x_test_index)

                        originalResultedLabel = myModel.predictOneItem(OriginalTrainedModel, x_test, x_test_index)

                        print(f'originalResultedLabel: {originalResultedLabel}')

                        # resultedLabel = myModel.predictOneItem(trainedModel, x_test, x_test_index, mrValue)

                        resultedLabel = myModel.predictOneItem(trainedModel, x_test, x_test_index)

                        print(f'resultedLabel: {resultedLabel}')

                        if not originalResultedLabel == resultedLabel: 

                            totalNotEqual = totalNotEqual + 1


                        else:
                            DoNothing = True
                            # print(f'#{total} they are equal')
                        
                        total = total + 1
                            # print(f'for x_test[{x_test_index}], originalResultedLabel = {originalResultedLabel} and resultedLabel = {resultedLabel}')
                            
                            # print('originalResultedLabel == resultedLabel', originalResultedLabel == resultedLabel)

                        # print('x_test_index: ', x_test_index)
                        # print('result: ', resultedLabel)


                    # for one_test in x_test:



                    # mrsAccuracyScores = mrsAccuracyScores + myModel.getModel(mrNumber, mrValue, fold_index, nfold)

    print('totalNotEqual: ', totalNotEqual)  
    print('total: ', total)  



getElementsComparison()


