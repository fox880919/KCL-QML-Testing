import numpy as np

# from default_parameters import DefaultParameters

from classes.default_parameters import DefaultParameters 

class MyParameters:

    useParametersClassParameters = True

    n_folds = 4
    # 0 = wine data
    # dataType = 0
    
    # 2 = creditcard
    # 3 = mnist
    dataType = 2

    #0 = amplitude embedding, 1 = angle embedding, 2 = custom embedding
    featureMapType = 0

    alwaysUsePCA = False 

    # pca_components = 8

    pca_components = 6


    applyMRs = True

    AskUserToApplyMRs = False

    askUserToInputParameters = False

    applyScalarValue = False
    scaleValue = 3

    fromScaleValue = 2

    # toScaleValue = 20
    toScaleValue = 5
    
    applyAngleRotation = False
    angle = 2* np.pi 

    applyPermutation = False

    invertAllLabels = False
    numberOfLabelsClasses = 3

    applyPerturbNoise = False
    perturbNoise = 0.1

    circuitDepth = 2
    applyCircuitDepth = False

    modifyCircuitDepth = False

    addAdditionalFeature = False

    addAdditionalInputsAndOutputs = False

    useTrainedModel= False

    useNoise= False

    modelName = 'svm00'
    
    allDataTypes=['Wine Data', 'Load Digits', 'Credit Card', 'MNIST']

    featureMaps=['Amplitude Embedding', 'Angle Embedding', 'Custom Embedding']

    # savingFileName = 'my_dataframe.csv'

    # savingFileName = 'my_dataframe_noisy.csv'

    savingFileName = 'my_dataframe_all_noise.csv'

    # savedModelsFolder = 'saved_models'

    # savedModelsFolder = 'saved_models_noisy'

    savedModelsFolder = 'saved_models_all_noise'

    applyDepolarizingChannelNoise = True

    applyAfterEnganglementNoise = False

    applyBitFlipNoise = False

    applyPhaseDumingNoise = False

    doOneKfold = False

    onlyKFoldValue = 0

    doOneScalar = False

    onlyScalarValue = 3

    usePercentageOfData = True

    PercentageOfData = 0.002

    def getModelName(mrNumber, mrValue, fold_index, n_folds):

        return 'SVM'+ str(0) + str(mrNumber)+ '-' + str(mrValue) + '-' + str(fold_index) + '-of-' + str(n_folds)

    def getFullPathModelName(modelName):

        return f'saved_models/{modelName}'


    def resetParameters():

        defaultParameters = DefaultParameters()

        MyParameters.useParametersClassParameters = defaultParameters.useParametersClassParameters

        MyParameters.n_folds = defaultParameters.n_folds
        # 0 = wine data
        # MyParameters.dataType = 0
        # 2 = creditcard
        MyParameters.dataType = defaultParameters.dataType


        #0 = amplitude embedding, 1 = angle embedding, 2 = custom embedding
        MyParameters.featureMapType = defaultParameters.featureMapType

        MyParameters.alwaysUsePCA = defaultParameters.alwaysUsePCA

        MyParameters.pca_components = defaultParameters.pca_components

        MyParameters.applyMRs = defaultParameters.applyMRs

        MyParameters.AskUserToApplyMRs = defaultParameters.AskUserToApplyMRs

        MyParameters.askUserToInputParameters = defaultParameters.askUserToInputParameters

        MyParameters.applyScalarValue = defaultParameters.applyScalarValue
        MyParameters.scaleValue = defaultParameters.scaleValue
        
        MyParameters.fromScaleValue = defaultParameters.fromScaleValue
        MyParameters.toScaleValue = defaultParameters.toScaleValue

        MyParameters.applyAngleRotation = defaultParameters.applyAngleRotation
        MyParameters.angle = defaultParameters.angle

        MyParameters.applyPermutation = defaultParameters.applyPermutation

        MyParameters.invertAllLabels = defaultParameters.invertAllLabels
        MyParameters.numberOfLabelsClasses = defaultParameters.numberOfLabelsClasses

        MyParameters.applyPerturbNoise = defaultParameters.applyPerturbNoise
        MyParameters.perturbNoise = defaultParameters.perturbNoise

        MyParameters.circuitDepth = defaultParameters.circuitDepth
        MyParameters.applyCircuitDepth = defaultParameters.applyCircuitDepth

        MyParameters.modifyCircuitDepth = defaultParameters.modifyCircuitDepth

        MyParameters.addAdditionalFeature = defaultParameters.addAdditionalFeature

        MyParameters.addAdditionalInputsAndOutputs = defaultParameters.addAdditionalFeature

        MyParameters.useTrainedModel= defaultParameters.useTrainedModel

        MyParameters.useNoise= defaultParameters.useNoise

        MyParameters.modelName = defaultParameters.modelName
        
        MyParameters.allDataTypes= defaultParameters.allDataTypes

        MyParameters.featureMaps= defaultParameters.featureMaps

        MyParameters.savingFileName = defaultParameters.savingFileName

        MyParameters.applyDepolarizingChannelNoise = defaultParameters.applyDepolarizingChannelNoise

        MyParameters.applyAfterEnganglementNoise = defaultParameters.applyAfterEnganglementNoise

        MyParameters.applyBitFlipNoise = defaultParameters.applyBitFlipNoise

        MyParameters.applyPhaseDumingNoise = defaultParameters.applyPhaseDumingNoise

        MyParameters.doOneKfold = defaultParameters.doOneKfold

        MyParameters.doOneScalar = defaultParameters.doOneScalar

        MyParameters.onlyScalarValue = defaultParameters.onlyScalarValue

        MyParameters.usePercentageOfData = defaultParameters.usePercentageOfData

        MyParameters.PercentageOfData = defaultParameters.PercentageOfData

        MyParameters.savingFileName = defaultParameters.savingFileName

        # MyParameters.savedModelsFolder = 'saved_models'

        MyParameters.savedModelsFolder = defaultParameters.savedModelsFolder
        


