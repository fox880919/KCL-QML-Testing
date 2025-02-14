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

# myParameters = MyParameters()

readUserInput = ReadUserInput()

useDefaultParameters = False

usedParameters = {'dataType': 1, 'featureMapType': 1, 'components': 8}

myMetamorphicTesting = MyMetamorphicTesting()

myMetamorphicTesting.myInit()

def DoUserWantToUseDefaultParameters():

    return readUserInput.checkIfUserWantsToUseDefaultParameters()

def DoUserWantToUseMRs():

    if MyParameters.AskUserToApplyMRs:

        message = "Should we apply MRs? (0 for Yes, and 1 for No)"

        applyMRs = readUserInput.readGenericBoleanInput(message)

        MyParameters.applyMRs = applyMRs

        return applyMRs
    else:
        return False

def checkImplementingPCA(x_tr, x_test):

    if not useDefaultParameters:

        featureMapType = readUserInput.readFeatureMapTypeInput()

    else: 

        featureMapType = MyParameters.featureMapType   

        print('default featureMapType is: ', featureMapType)

        usedParameters['featureMapType'] = featureMapType

        # return;

    components = 0

    if featureMapType == 1:

        defaultNumber = MyParameters.pca_components

        message = "Enter components number for PCA (default: {defaultNumber}):"
        
        if not useDefaultParameters:

            components = readUserInput.readGeneralNumericInput(message, defaultNumber)
        else:

            components = MyParameters.pca_components   

            print('default components number is: ', components)

        myPCA = MyPCA()

        x_tr, x_test = myPCA.implementPCA(x_tr, x_test, components)

        MyParameters.pca_components = components

        if(MyParameters.featureMapType == 1):

            print('applied PCA with components number:', components)


    usedParameters['components'] = components


    return x_tr, x_test

def getData():

    if not useDefaultParameters:
        dataType = readUserInput.readDataTypeInput()
    else:

        dataType = MyParameters.dataType

        print('default dataType is: ', dataType)

    usedParameters['dataType'] = dataType

    np, x_tr, x_test, y_tr, y_test = dataManager.getDatabyNumber(dataType)

    # print('original x_tr: ', x_tr[0])

    # if MyParameters.applyAngleRotation:

        # x_tr, x_test = checkImplementingPCA(x_tr, x_test)

    x_tr, x_test = checkImplementingPCA(x_tr, x_test)

    # print('after PCA x_tr: ', x_tr[0])

    
    #1
    if MyParameters.applyScalarValue:
        x_tr, x_test = myMetamorphicTesting.scaleInputData(x_tr, x_test, MyParameters.scaleValue)
        print('scaled x_tr and x_test by:', MyParameters.scaleValue)

    
    #2
    if MyParameters.applyAngleRotation:

        # print('before rotation')
        # print(x_test)


        x_tr = myMetamorphicTesting.rotateInputDataWithAngle(x_tr, MyParameters.angle)
        x_test = myMetamorphicTesting.rotateInputDataWithAngle(x_test, MyParameters.angle)

        # print('after rotation x_tr: ', x_tr[0])

        print('rotated x_tr and x_test with angle ', MyParameters.angle)

        # print('after rotation')
        # print(x_test)


    #3
    if MyParameters.applyPermutation:
        x_tr, y_tr= myMetamorphicTesting.permutateInputData(x_tr, y_tr)

        print('permutated both of x_tr and y_tr')


    #4
    if MyParameters.invertAllLabels:
        y_tr, y_test= myMetamorphicTesting.invertAllLabels(y_tr, y_test, MyParameters.numberOfLabelsClasses)

        print('inverted labels of both of y_tr and y_test')


    #5
    if MyParameters.applyPerturbNoise:
        x_test = myMetamorphicTesting.perturbParameters(x_test, MyParameters.perturbNoise)

        print('applyed noise by {delta} to x_test')

    

    #6
    if MyParameters.modifyCircuitDepth:
        myMetamorphicTesting.modifyCircuitDepth(MyParameters.featureMapType)


    # print('MyParameters.addAdditionalFeature: ', MyParameters.addAdditionalFeature)

    #7
    if MyParameters.addAdditionalFeature:
        
        # print('before x_tr[0]: ', x_tr[0])
        x_tr, x_test = myMetamorphicTesting.addingAdditionalFeature(x_tr, x_test)

        print('MyParameters.pca_components: ', MyParameters.pca_components)
        # print('after addAdditionalFeature x_tr.shape: ', x_tr.shape)
        # print('after addAdditionalFeature x_test.shape: ', x_test.shape)
        # print('after x_tr[0]: ', x_tr[0])

    #8
    if MyParameters.addAdditionalInputsAndOutputs:
    
        x_tr, x_test, y_tr, y_test  = myMetamorphicTesting.addingRedundantInputsAndOutputs(x_tr, x_test, y_tr, y_test)

        # print('after addAdditionalInputsAndOutputs x_tr.shape: ', x_tr.shape)
        # print('after addAdditionalInputsAndOutputs x_test.shape: ', x_test.shape)
        # print('after addAdditionalInputsAndOutputs y_tr.shape: ', y_tr.shape)
        # print('after addAdditionalInputsAndOutputs y_test.shape: ', y_test.shape)
 

    # print('x_tr: ', x_tr)

    # if not MyParameters.applyAngleRotation:

    #     x_tr, x_test = checkImplementingPCA(x_tr, x_test)


    return np, x_tr, x_test, y_tr, y_test


def useQSVM(np, x_tr, x_test, y_tr, y_test):

    myKernel = MyKernel()

    myAccuracyScore = myKernel.startSVC(np, x_tr, x_test, y_tr, y_test, MyParameters.featureMapType, MyParameters.pca_components)

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

    # MyParameters = newParameters
 
    np, x_tr, x_test, y_tr, y_test = getData()

    print('data is prepared')

    myAccuracyScore = useQSVM(np, x_tr, x_test, y_tr, y_test)

    print('myAccuracyScore: ', myAccuracyScore)
    saveToDataFrame(myAccuracyScore, usedMetaMorphic = False)

    print('process is finished')
# nqubits, input_tr, input_test, output_tr, output_test = useQSVMWithMetamorphic()

def tryManyParameters():

    print('trying many parameters')
    # #0
    # MyParameters.featureMapType = 0
    # runMain()

    # MyParameters.featureMapType = 1
    # MyParameters.pca_components = 8
    # runMain()
    # MyParameters.pca_components = 13
    
    # MyParameters.featureMapType = 2
    # runMain()
    
    # #1
    # MyParameters.featureMapType = 0
    # MyParameters.applyScalarValue = True
    # runMain()
    # MyParameters.applyScalarValue = False

    # #2
    # MyParameters.featureMapType = 1
    # MyParameters.pca_components = 8
    # MyParameters.applyAngleRotation = True
    # runMain()
    # MyParameters.pca_components = 13
    # MyParameters.applyAngleRotation = False

    # MyParameters.featureMapType = 2
    # MyParameters.applyAngleRotation = True
    # runMain()
    # MyParameters.applyAngleRotation = False

    # #3
    # MyParameters.featureMapType = 0
    # MyParameters.applyPermutation = True
    # runMain()
    # MyParameters.applyPermutation = False

    # MyParameters.featureMapType = 1
    # MyParameters.pca_components = 8
    # MyParameters.applyPermutation = True
    # runMain()
    # MyParameters.pca_components = 13
    # MyParameters.applyPermutation = False

    # MyParameters.featureMapType = 2
    # MyParameters.applyPermutation = True
    # runMain()
    # MyParameters.applyPermutation = False

    # #4
    # MyParameters.featureMapType = 0
    # MyParameters.invertAllLabels = True
    # runMain()
    # MyParameters.invertAllLabels = False

    # MyParameters.featureMapType = 1
    # MyParameters.pca_components = 8
    # MyParameters.invertAllLabels = True
    # runMain()
    # MyParameters.pca_components = 13
    # MyParameters.invertAllLabels = False

    # MyParameters.featureMapType = 2
    # MyParameters.invertAllLabels = True
    # runMain()
    # MyParameters.invertAllLabels = False

    # #7
    # MyParameters.featureMapType = 0
    # MyParameters.addAdditionalFeature = True
    # runMain()
    # MyParameters.addAdditionalFeature = False

    MyParameters.featureMapType = 1
    MyParameters.pca_components = 6
    MyParameters.addAdditionalFeature = True
    runMain()
    MyParameters.pca_components = 13
    MyParameters.addAdditionalFeature = False

    # MyParameters.featureMapType = 2
    # MyParameters.addAdditionalFeature = True
    # runMain()
    # MyParameters.addAdditionalFeature = False

    # #8
    # MyParameters.featureMapType = 0
    # MyParameters.addAdditionalInputsAndOutputs = True
    # runMain()
    # MyParameters.addAdditionalFeature = False

    # MyParameters.featureMapType = 1
    # MyParameters.pca_components = 8
    # MyParameters.addAdditionalInputsAndOutputs = True
    # runMain()
    # MyParameters.pca_components = 13
    # MyParameters.addAdditionalFeature = False

    # MyParameters.featureMapType = 2
    # MyParameters.addAdditionalInputsAndOutputs = True
    # runMain()
    # MyParameters.addAdditionalFeature = False

    #9 PCA with components = 13
    # MyParameters.featureMapType = 1
    # MyParameters.pca_components = 13
    # print('MyParameters.pca_components: ', MyParameters.pca_components)
    # MyParameters.applyAngleRotation = True
    # runMain()
    # MyParameters.pca_components = 13
    # MyParameters.applyAngleRotation = False





if MyParameters.askUserToInputParameters:

    useDefaultParameters = DoUserWantToUseDefaultParameters()
else:
    useDefaultParameters = True


if MyParameters.AskUserToApplyMRs:

    useMetamorphicRelations = DoUserWantToUseMRs()

tryManyParameters()



