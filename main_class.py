import sys

from classes.time import MyTimeHelper

from datetime import datetime


sys.path.insert(0, './data')
sys.path.insert(1, './metamorphic')
sys.path.insert(2, './classes')
sys.path.insert(3, './qsvm')
sys.path.insert(4, './metamorphic')

from classes.parameters import MyParameters

from classes.read_user_input import ReadUserInput

from data.data_manager import DataManager

from qsvm.my_kernel import MyKernel

from qsvm.my_qsvm import MyQSVM

from data.my_dataframe import MyDataFrame

from metamorphic.my_metamorphic_testing import MyMetamorphicTesting

from qsvm.my_pca import MyPCA

class MyMain():

    dataManager = DataManager()

    # myParameters = MyParameters()

    readUserInput = ReadUserInput()

    useDefaultParameters = True

    usedParameters = {'dataType': 1, 'featureMapType': 1, 'components': 8}

    myMetamorphicTesting = MyMetamorphicTesting()

    myMetamorphicTesting.myInit()


    def DoUserWantToUseDefaultParameters(self):

        return MyMain.readUserInput.checkIfUserWantsToUseDefaultParameters()

    def DoUserWantToUseMRs(self ):

        if MyParameters.AskUserToApplyMRs:

            message = "Should we apply MRs? (0 for Yes, and 1 for No)"

            applyMRs = MyMain.readUserInput.readGenericBoleanInput(message)

            MyParameters.applyMRs = applyMRs

            return applyMRs
        else:
            return False
        
    def checkImplementingPCA(x_tr, x_test):

        if not MyMain.useDefaultParameters:

            featureMapType = MyMain.readUserInput.readFeatureMapTypeInput()

        else: 

            featureMapType = MyParameters.featureMapType   

            # print('default featureMapType is: ', featureMapType)

            MyMain.usedParameters['featureMapType'] = featureMapType

            # return;

        components = 0

        print(f'featureMapType: {featureMapType}')
        if featureMapType == 1:

            print('in featureMapType == 1')

            # return x_tr, x_test
            defaultNumber = MyParameters.pca_components

            message = "Enter components number for PCA (default: {defaultNumber}):"
            
            if not MyMain.useDefaultParameters:

                components = MyMain.readUserInput.readGeneralNumericInput(message, defaultNumber)
            else:

                components = MyParameters.pca_components   

                print('default components number is: ', components)

            myPCA = MyPCA()

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



        MyMain.usedParameters['components'] = components


        return x_tr, x_test
    
    # def getData():
    def getListOfFoldData():


        if not MyMain.useDefaultParameters:
            dataType = MyMain.readUserInput.readDataTypeInput()
        else:

            dataType = MyParameters.dataType

        print('default dataType is: ', dataType)

        MyMain.usedParameters['dataType'] = dataType

        np, train_data_list, test_data_list = MyMain.dataManager.getListOfFoldDatabyNumber(dataType)

        return np, train_data_list, test_data_list
    
    def dataModification(x_tr, x_test, y_tr, y_test):

        # print('before pca, x_tr[0]: ', x_tr[0])
        print('before pca, len(x_tr[0]): ', len(x_tr[0]))

        x_tr, x_test = MyMain.checkImplementingPCA(x_tr, x_test)

        # print('after pca x_tr[0]: ', x_tr[0])
        print('after pca len(x_tr[0]: ', len(x_tr[0]))

        #1
        if MyParameters.applyScalarValue:
            x_tr, x_test = MyMain.myMetamorphicTesting.scaleInputData(x_tr, x_test, MyParameters.scaleValue)
            # print('scaled x_tr and x_test by:', MyParameters.scaleValue)
    
        #2
        if MyParameters.applyAngleRotation:

            x_tr = MyMain.myMetamorphicTesting.rotateInputDataWithAngle(x_tr, MyParameters.angle)
            x_test = MyMain.myMetamorphicTesting.rotateInputDataWithAngle(x_test, MyParameters.angle)

            # print('rotated x_tr and x_test with angle ', MyParameters.angle)

        #3
        if MyParameters.applyPermutation:
            x_tr, y_tr= MyMain.myMetamorphicTesting.permutateInputData(x_tr, y_tr)

            # print('permutated both of x_tr and y_tr')

        #4
        if MyParameters.invertAllLabels:
            y_tr, y_test= MyMain.myMetamorphicTesting.invertAllLabels(y_tr, y_test, MyParameters.numberOfLabelsClasses)

            # print('inverted labels of both of y_tr and y_test')

        #5
        if MyParameters.applyPerturbNoise:
            x_test = MyMain.myMetamorphicTesting.perturbParameters(x_test, MyParameters.perturbNoise)

            # print('applyed noise by {delta} to x_test')

        #6
        if MyParameters.modifyCircuitDepth:
            MyMain.myMetamorphicTesting.modifyCircuitDepth(MyParameters.featureMapType)

        #7
        if MyParameters.addAdditionalFeature:
            
            x_tr, x_test = MyMain.myMetamorphicTesting.addingAdditionalFeature(x_tr, x_test)

            # print('MyParameters.pca_components: ', MyParameters.pca_components)

        #8
        if MyParameters.addAdditionalInputsAndOutputs:
        
            x_tr, x_test, y_tr, y_test  = MyMain.myMetamorphicTesting.addingRedundantInputsAndOutputs(x_tr, x_test, y_tr, y_test)

        # x_tr, x_test = MyMain.checkImplementingPCA(x_tr, x_test)

        return  x_tr, x_test, y_tr, y_test
    

    def useQSVM(np, x_tr, x_test, y_tr, y_test, modelName, fold_index):


        myQSVM = MyQSVM()

        myAccuracyScore = myQSVM.startSVC(np, x_tr, x_test, y_tr, y_test,  modelName, fold_index, MyParameters.featureMapType, MyParameters.pca_components)

        return myAccuracyScore

    
    def useMR(self, mrNumber, mrValue):

        
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

        np, train_data_list, test_data_list = MyMain.getListOfFoldData()

        # print('len(train_data_list) = ', len(train_data_list))
        # print('len(train_data_list[0]) = ', len(train_data_list[0]))

        print('len(train_data_list[0][0]) = ', len(train_data_list[0][0]))

        # print('train_data_list = ', train_data_list)

        # return;
    
        for fold_index in range(len(train_data_list)):

            # print(f'checking MyParameters.doOneKfold == {MyParameters.doOneKfold}')

            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


            # if fold_index == 0:
            #     continue

            print(f'starttime of kfold({fold_index}): {time}')
            
            # print(f'checking {fold_index} == {MyParameters.onlyKFoldValue}: {fold_index == MyParameters.onlyKFoldValue}')
            #to do one kfold and skip others if wanted
            if MyParameters.doOneKfold == True:
                # if fold_index != MyParameters.onlyKFoldValue:
                if fold_index != MyParameters.onlyKFoldValue:
            
                    continue

            x_tr, y_tr = train_data_list[fold_index]
            x_test, y_test = test_data_list[fold_index]

            print('fold_index: ', fold_index)
            # return
            x_tr, x_test, y_tr, y_test = MyMain.dataModification(x_tr, x_test, y_tr, y_test)

            # modelName = 'SVM'+ str(0) + str(mrNumber)+ '-' + str(mrValue) + '-' + str(fold_index) + '-of-' + str(MyParameters.n_folds)

            modelName = MyParameters.getModelName(mrNumber, mrValue, fold_index, MyParameters.n_folds)

            print('modelName is: ', modelName)

            myAccuracyScore = MyMain.useQSVM(np, x_tr, x_test, y_tr, y_test, modelName, fold_index)

            MyMain.saveToDataFrame(myAccuracyScore, usedMetaMorphic, fold_index)
            
            print(f'endtime of kfold({fold_index}): {time}')



    def saveToDataFrame(myAccuracyScore, usedMetaMorphic, foldIndex):
        
        dateAndTime = MyTimeHelper().getTimeNow()
        
        myDataFrame = MyDataFrame()

        formattedData = myDataFrame.formatData(myAccuracyScore, usedMetaMorphic, MyMain.usedParameters, dateAndTime, foldIndex)

        print('formattedData: ', formattedData)

        myDataFrame.processToDataFrame(formattedData)
