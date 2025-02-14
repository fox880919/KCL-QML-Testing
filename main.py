import sys

from classes.time import MyTimeHelper

sys.path.insert(0, './data')
sys.path.insert(1, './metamorphic')
sys.path.insert(2, './classes')
sys.path.insert(3, './qsvm')
sys.path.insert(4, './metamorphic')

from classes.parameters import MyParameters

from classes.read_user_input import ReadUserInput

from data.data_manager import DataManager

from qsvm.my_kernel import MyKernel

from data.my_dataframe import MyDataFrame

from metamorphic.my_metamorphic_testing import MyMetamorphicTesting

from qsvm.my_pca import MyPCA


dataManager = DataManager()

myParameters = MyParameters()

readUserInput = ReadUserInput()

useDefaultParameters = False

usedParameters = {'dataType': 1, 'featureMapType': 1, 'components': 8}

myMetamorphicTesting = MyMetamorphicTesting()

myMetamorphicTesting.myInit()

def DoUserWantToUseDefaultParameters():

    return readUserInput.checkIfUserWantsToUseDefaultParameters()

def DoUserWantToUseMRs():

    if myParameters.AskUserToApplyMRs:

        message = "Should we apply MRs? (0 for Yes, and 1 for No)"

        applyMRs = readUserInput.readGenericBoleanInput(message)

        myParameters.applyMRs = applyMRs

        return applyMRs
    else:
        return False

def checkImplementingPCA(x_tr, x_test):

    if not useDefaultParameters:

        featureMapType = readUserInput.readFeatureMapTypeInput()

    else: 

        featureMapType = myParameters.featureMapType   

        print('default featureMapType is: ', featureMapType)

        usedParameters['featureMapType'] = featureMapType

        # return;

    components = 0

    if featureMapType == 1:

        defaultNumber = myParameters.pca_components

        message = "Enter components number for PCA (default: {defaultNumber}):"
        
        if not useDefaultParameters:

            components = readUserInput.readGeneralNumericInput(message, defaultNumber)
        else:

            components = myParameters.pca_components   

            print('default components number is: ', components)

        myPCA = MyPCA()

        x_tr, x_test = myPCA.implementPCA(x_tr, x_test, components)

        MyParameters.pca_components = components

        if(myParameters.featureMapType == 1):

            print('applied PCA with components number:', components)


    usedParameters['components'] = components


    return x_tr, x_test

def getData():

    if not useDefaultParameters:
        dataType = readUserInput.readDataTypeInput()
    else:

        dataType = myParameters.dataType

        print('default dataType is: ', dataType)

    usedParameters['dataType'] = dataType

    np, x_tr, x_test, y_tr, y_test = dataManager.getDatabyNumber(dataType)

    # print('original x_tr: ', x_tr[0])

    x_tr, x_test = checkImplementingPCA(x_tr, x_test)

    # print('after PCA x_tr: ', x_tr[0])

    
    #1
    if myParameters.applyScalarValue:
        x_tr, x_test = myMetamorphicTesting.scaleInputData(x_tr, x_test, myParameters.scaleValue)
        print('scaled x_tr and x_test by:', myParameters.scaleValue)

    
    #2
    if myParameters.applyAngleRotation:

        # print('before rotation')
        # print(x_test)


        x_tr = myMetamorphicTesting.rotateInputDataWithAngle(x_tr, myParameters.angle)
        x_test = myMetamorphicTesting.rotateInputDataWithAngle(x_test, myParameters.angle)

        # print('after rotation x_tr: ', x_tr[0])

        print('rotated x_tr and x_test with angle ', myParameters.angle)

        # print('after rotation')
        # print(x_test)


    #3
    if myParameters.applyPermutation:
        x_tr, y_tr= myMetamorphicTesting.permutateInputData(x_tr, y_tr)

        print('permutated both of x_tr and y_tr')


    #4
    if myParameters.invertAllLabels:
        y_tr, y_test= myMetamorphicTesting.invertAllLabels(y_tr, y_test, myParameters.numberOfLabelsClasses)

        print('inverted labels of both of y_tr and y_test')


    #5
    if myParameters.applyPerturbNoise:
        x_test = myMetamorphicTesting.perturbParameters(x_test, myParameters.perturbNoise)

        print('applyed noise by {delta} to x_test')

    

    #6
    if myParameters.modifyCircuitDepth:
        myMetamorphicTesting.modifyCircuitDepth(myParameters.featureMapType)


    # print('myParameters.addAdditionalFeature: ', myParameters.addAdditionalFeature)

    #7
    if myParameters.addAdditionalFeature:
        
        # print('before x_tr[0]: ', x_tr[0])
        x_tr, x_test = myMetamorphicTesting.addingAdditionalFeature(x_tr, x_test)

        # print('after addAdditionalFeature x_tr.shape: ', x_tr.shape)
        # print('after addAdditionalFeature x_test.shape: ', x_test.shape)
        # print('after x_tr[0]: ', x_tr[0])

    #8
    if myParameters.addAdditionalInputsAndOutputs:
    
        x_tr, x_test, y_tr, y_test  = myMetamorphicTesting.addingRedundantInputsAndOutputs(x_tr, x_test, y_tr, y_test)

        # print('after addAdditionalInputsAndOutputs x_tr.shape: ', x_tr.shape)
        # print('after addAdditionalInputsAndOutputs x_test.shape: ', x_test.shape)
        # print('after addAdditionalInputsAndOutputs y_tr.shape: ', y_tr.shape)
        # print('after addAdditionalInputsAndOutputs y_test.shape: ', y_test.shape)
 

    # print('x_tr: ', x_tr)

    return np, x_tr, x_test, y_tr, y_test


def useQSVM(np, x_tr, x_test, y_tr, y_test):

    myKernel = MyKernel()

    myAccuracyScore = myKernel.startSVC(np, x_tr, x_test, y_tr, y_test, myParameters.featureMapType, myParameters.pca_components)

    return myAccuracyScore

def useQSVMWithMetamorphic():
    myMetamorphicTesting = MyMetamorphicTesting()
    myMetamorphicTesting.myInit()
    input_tr, input_test, output_tr, output_test = myMetamorphicTesting.pickTypeAndPrepareData(type)
    # nqubits, kernel_cerc = myMetamorphicTesting.getCiruitDetails()
    nqubits = myMetamorphicTesting.getCiruitDetails()

    return nqubits, input_tr, input_test, output_tr, output_test

def saveToDataFrame(myAccuracyScore, usedMetaMorphic):
    
    dateAndTime = MyTimeHelper().getTimeNow()
    
    myDataFrame = MyDataFrame()

    formattedData = myDataFrame.formatData(myAccuracyScore, usedMetaMorphic, usedParameters, dateAndTime)

    print('formattedData: ', formattedData)

    myDataFrame.processToDataFrame(formattedData)
    

def runMain():

    # myParameters = newParameters
 
    np, x_tr, x_test, y_tr, y_test = getData()

    print('data is prepared')

    myAccuracyScore = useQSVM(np, x_tr, x_test, y_tr, y_test)

    print('myAccuracyScore: ', myAccuracyScore)
    saveToDataFrame(myAccuracyScore, usedMetaMorphic = False)

    print('process is finished')
# nqubits, input_tr, input_test, output_tr, output_test = useQSVMWithMetamorphic()

def tryManyParameters():

    myParameters.featureMapType = 0
    runMain()

    myParameters.featureMapType = 1
    myParameters.pca_components = 8
    runMain()
    myParameters.pca_components = 13
    
    myParameters.featureMapType = 2
    runMain()
    
    #1
    myParameters.featureMapType = 0
    myParameters.applyScalarValue = True
    runMain()
    myParameters.applyScalarValue = False

    #2
    myParameters.featureMapType = 1
    myParameters.pca_components = 8
    myParameters.applyAngleRotation = True
    runMain()
    myParameters.pca_components = 13
    myParameters.applyAngleRotation = False

    myParameters.featureMapType = 2
    myParameters.applyAngleRotation = True
    runMain()
    myParameters.applyAngleRotation = False

    #3
    myParameters.featureMapType = 0
    myParameters.applyPermutation = True
    runMain()
    myParameters.applyPermutation = False

    myParameters.featureMapType = 1
    myParameters.pca_components = 8
    myParameters.applyPermutation = True
    runMain()
    myParameters.pca_components = 13
    myParameters.applyPermutation = False

    myParameters.featureMapType = 2
    myParameters.applyPermutation = True
    runMain()
    myParameters.applyPermutation = False

    #4
    myParameters.featureMapType = 0
    myParameters.invertAllLabels = True
    runMain()
    myParameters.invertAllLabels = False

    myParameters.featureMapType = 1
    myParameters.pca_components = 8
    myParameters.invertAllLabels = True
    runMain()
    myParameters.pca_components = 13
    myParameters.invertAllLabels = False

    myParameters.featureMapType = 2
    myParameters.invertAllLabels = True
    runMain()
    myParameters.invertAllLabels = False

    #7
    myParameters.featureMapType = 0
    myParameters.addAdditionalFeature = True
    runMain()
    myParameters.addAdditionalFeature = False

    myParameters.featureMapType = 1
    myParameters.pca_components = 8
    myParameters.addAdditionalFeature = True
    runMain()
    myParameters.pca_components = 13
    myParameters.addAdditionalFeature = False

    myParameters.featureMapType = 2
    myParameters.addAdditionalFeature = True
    runMain()
    myParameters.addAdditionalFeature = False

    #8
    myParameters.featureMapType = 0
    myParameters.addAdditionalInputsAndOutputs = True
    runMain()
    myParameters.addAdditionalFeature = False

    myParameters.featureMapType = 1
    myParameters.pca_components = 8
    myParameters.addAdditionalInputsAndOutputs = True
    runMain()
    myParameters.pca_components = 13
    myParameters.addAdditionalFeature = False

    myParameters.featureMapType = 2
    myParameters.addAdditionalInputsAndOutputs = True
    runMain()
    myParameters.addAdditionalFeature = False



if myParameters.askUserToInputParameters:

    useDefaultParameters = DoUserWantToUseDefaultParameters()
else:
    useDefaultParameters = True


if myParameters.AskUserToApplyMRs:

    useMetamorphicRelations = DoUserWantToUseMRs()

tryManyParameters()



