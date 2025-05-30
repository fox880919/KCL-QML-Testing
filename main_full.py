
import sys

sys.path.insert(0, './data')
sys.path.insert(1, './metamorphic')
sys.path.insert(2, './classes')
sys.path.insert(3, './qsvm')
sys.path.insert(4, './metamorphic')


from classes.parameters import MyParameters

from datetime import datetime

from classes.time import MyTimeHelper

from data.my_dataframe import MyDataFrame

from data.data_manager import DataManager

from metamorphic.my_metamorphic_testing import MyMetamorphicTesting

from qsvm.my_qsvm import MyQSVM

from qsvm.my_pca import MyPCA

from qsvm.my_feature_map import MyFeatureMap


dataManager = DataManager()

usedParameters = {'dataType': 1, 'featureMapType': 1, 'components': 8}

myMetamorphicTesting = MyMetamorphicTesting()

myQSVM = MyQSVM()

myPCA = MyPCA()

def startSVC(np, input_tr, input_test, output_tr, output_test, modelName, fold_index, featureMapType = 0, components = 8):
        
        MyQSVM.np = np

        # fullModelName = 'saved_models/' + modelName
        fullModelName = modelName

        myFeatureMap = MyFeatureMap()

        MyQSVM.myKernel.np = MyQSVM.np

        print(f'before feature map, backend: {MyParameters.backend}')

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

def useQSVM(np, x_tr, x_test, y_tr, y_test, modelName, fold_index):

        myAccuracyScore = startSVC(np, x_tr, x_test, y_tr, y_test,  modelName, fold_index, MyParameters.featureMapType, MyParameters.pca_components)

        return myAccuracyScore


def dataModification(x_tr, x_test, y_tr, y_test):

        # print('before pca, x_tr[0]: ', x_tr[0])
        print('before pca, len(x_tr[0]): ', len(x_tr[0]))

        x_tr, x_test = checkImplementingPCA(x_tr, x_test)

        # print('after pca x_tr[0]: ', x_tr[0])
        print('after pca len(x_tr[0]: ', len(x_tr[0]))

        #1
        if MyParameters.applyScalarValue:
            x_tr, x_test = myMetamorphicTesting.scaleInputData(x_tr, x_test, MyParameters.scaleValue)
            # print('scaled x_tr and x_test by:', MyParameters.scaleValue)

        return  x_tr, x_test, y_tr, y_test


def checkImplementingPCA(x_tr, x_test):

        featureMapType = MyParameters.featureMapType   

        # print('default featureMapType is: ', featureMapType)

        usedParameters['featureMapType'] = featureMapType

            # return;

        components = 0

        print(f'featureMapType: {featureMapType}')
        if featureMapType == 1:

            print('in featureMapType == 1')

            # return x_tr, x_test
            defaultNumber = MyParameters.pca_components

            message = "Enter components number for PCA (default: {defaultNumber}):"

            components = MyParameters.pca_components   

            print('default components number is: ', components)

            x_tr, x_test = myPCA.implementPCA(x_tr, x_test, components)

            MyParameters.pca_components = components

            if(MyParameters.featureMapType == 1):

                print('applied PCA with components number:', components)
        else: 
            
            if MyParameters.alwaysUsePCA:
                
                myPCA = MyPCA()

                x_tr, x_test = myPCA.implementPCA(x_tr, x_test, MyParameters.pca_components)
            
            else:
                print('no PCA applied')

        usedParameters['components'] = components

        return x_tr, x_test

def saveToDataFrame(myAccuracyScore, usedMetaMorphic, foldIndex, startTime ='', endTime = ''):
        
        dateAndTime = MyTimeHelper().getTimeNow()
        
        myDataFrame = MyDataFrame()

        formattedData = myDataFrame.formatData(myAccuracyScore, usedMetaMorphic, usedParameters, dateAndTime,
                                                foldIndex, startTime, endTime)

        print('formattedData: ', formattedData)

        myDataFrame.processToDataFrame(formattedData)

def getListOfFoldData():

        dataType = MyParameters.dataType

        print('default dataType is: ', dataType)

        usedParameters['dataType'] = dataType

        np, train_data_list, test_data_list = dataManager.getListOfFoldDatabyNumber(dataType)

        return np, train_data_list, test_data_list

def useMR(mrNumber, mrValue):

        if not mrNumber == 0:

            print('reset parameters')
            MyParameters.resetParameters()

        usedMetaMorphic = True

        if mrNumber == 1:
            
            MyParameters.applyScalarValue = True

            MyParameters.scaleValue = mrValue

            #to do one scalar value and skip others if wanted
            if MyParameters.doOneScalar == True:
                if mrValue != MyParameters.onlyScalarValue:
                    return;    

        elif mrNumber == 2:

            MyParameters.featureMapType = 1
            
            MyParameters.applyAngleRotation = True

            MyParameters.angle = mrValue

        elif mrNumber == 3:
             MyParameters.applyPermutation = True

        elif mrNumber == 4:
            MyParameters.invertAllLabels = True

        elif mrNumber == 5:
            
            MyParameters.applyPerturbNoise = True   

            MyParameters.perturbNoise = mrValue

        else:
            
            usedMetaMorphic = False
            print('no change')

        np, train_data_list, test_data_list = getListOfFoldData()

        print('kfolds = len(train_data_list) = ', len(train_data_list))

        print('inputs per kfold = len(train_data_list[0][0]) = ', len(train_data_list[0][0]))

        print('input dimensions len(train_data_list[0][0][0]) = ', len(train_data_list[0][0][0]))
    
        for fold_index in range(len(train_data_list)):

            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print(f'starttime of kfold({fold_index}): {time}')
            if MyParameters.doOneKfold == True:
                if fold_index != MyParameters.onlyKFoldValue:
            
                    continue

            x_tr, y_tr = train_data_list[fold_index]
            x_test, y_test = test_data_list[fold_index]

            print('fold_index: ', fold_index)
            # return
            x_tr, x_test, y_tr, y_test = dataModification(x_tr, x_test, y_tr, y_test)

            # modelName = 'SVM'+ str(0) + str(mrNumber)+ '-' + str(mrValue) + '-' + str(fold_index) + '-of-' + str(MyParameters.n_folds)

            modelName = MyParameters.getModelName(mrNumber, mrValue, fold_index, MyParameters.n_folds)

            print('modelName is: ', modelName)

            startTime = MyTimeHelper().getTimeNow()

            print(f'startTime: {startTime}')
            myAccuracyScore = useQSVM(np, x_tr, x_test, y_tr, y_test, modelName, fold_index)

            endTime =  MyTimeHelper().getTimeNow()
            print(f'endTime: {startTime}')

            saveToDataFrame(myAccuracyScore, usedMetaMorphic, fold_index, startTime, endTime)
            
            print(f'endtime of kfold({fold_index}): {time}')


def loopThroughParametersForMRs(mrNumber, value):
    
    roundMessage = 'starting round of '+ str(mrNumber) + ' and value ' + str(value)
    print(roundMessage)

    useMR(mrNumber, value)


def start():
    for i in range(0, 6):

        if i == 0:

            loopThroughParametersForMRs(i, 0)

        if i == 1:

            for j in range(MyParameters.fromScaleValue, MyParameters.toScaleValue):

                loopThroughParametersForMRs(i, j)

start()