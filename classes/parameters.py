import numpy as np

# from default_parameters import DefaultParameters

from classes.default_parameters import DefaultParameters 

class MyParameters:

    useParametersClassParameters = DefaultParameters.useParametersClassParameters

    n_folds = DefaultParameters.n_folds
    # 0 = wine data
    # dataType = 0
    
    # 2 = creditcard
    # 3 = mnist
    dataType = DefaultParameters.dataType

    #0 = amplitude embedding, 1 = angle embedding, 2 = custom embedding
    featureMapType = DefaultParameters.featureMapType

    alwaysUsePCA = DefaultParameters.alwaysUsePCA 

    # pca_components = 8

    pca_components = DefaultParameters.pca_components


    applyMRs = DefaultParameters.applyMRs

    AskUserToApplyMRs = DefaultParameters.AskUserToApplyMRs

    askUserToInputParameters = DefaultParameters.askUserToInputParameters

    applyScalarValue = DefaultParameters.applyScalarValue
    scaleValue = DefaultParameters.scaleValue

    fromScaleValue = DefaultParameters.fromScaleValue

    # toScaleValue = 20
    toScaleValue = DefaultParameters.toScaleValue
    
    applyAngleRotation = DefaultParameters.applyAngleRotation
    angle = DefaultParameters.angle

    applyPermutation = DefaultParameters.applyPermutation

    invertAllLabels = DefaultParameters.invertAllLabels
    numberOfLabelsClasses = DefaultParameters.numberOfLabelsClasses

    applyPerturbNoise = DefaultParameters.applyPerturbNoise
    perturbNoise = DefaultParameters.perturbNoise

    circuitDepth = DefaultParameters.circuitDepth
    applyCircuitDepth = DefaultParameters.applyCircuitDepth

    modifyCircuitDepth = DefaultParameters.modifyCircuitDepth

    addAdditionalFeature = DefaultParameters.addAdditionalFeature

    addAdditionalInputsAndOutputs = DefaultParameters.addAdditionalInputsAndOutputs

    useTrainedModel= DefaultParameters.useTrainedModel

    useNoise= DefaultParameters.useNoise

    modelName = DefaultParameters.modelName
    
    allDataTypes= DefaultParameters.allDataTypes

    featureMaps= DefaultParameters.featureMaps

    # savingFileName = 'my_dataframe.csv'

    # savingFileName = 'my_dataframe_noisy.csv'

    savingFileName = DefaultParameters.savingFileName

    # savedModelsFolder = 'saved_models'

    # savedModelsFolder = 'saved_models_noisy'

    savedModelsFolder = DefaultParameters.savedModelsFolder

    applyDepolarizingChannelNoise = DefaultParameters.applyDepolarizingChannelNoise

    depolarizingChannelNoise = DefaultParameters.depolarizingChannelNoise

    applyAfterEnganglementNoise = DefaultParameters.applyAfterEnganglementNoise

    afterEnganglementNoise = DefaultParameters.afterEnganglementNoise

    applyBitFlipNoise = DefaultParameters.applyBitFlipNoise

    bitFlipNoise = DefaultParameters.bitFlipNoise

    applyPhaseDumingNoise = DefaultParameters.applyPhaseDumingNoise

    phaseDumingNoise = DefaultParameters.phaseDumingNoise

    doOneKfold = DefaultParameters.doOneKfold

    onlyKFoldValue = DefaultParameters.onlyKFoldValue

    doOneScalar = DefaultParameters.doOneScalar

    onlyScalarValue = DefaultParameters.onlyScalarValue

    usePercentageOfData = DefaultParameters.usePercentageOfData

    PercentageOfData =DefaultParameters.PercentageOfData

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

        MyParameters.depolarizingChannelNoise = defaultParameters.depolarizingChannelNoise

        MyParameters.applyAfterEnganglementNoise = defaultParameters.applyAfterEnganglementNoise

        MyParameters.afterEnganglementNoise = defaultParameters.afterEnganglementNoise

        MyParameters.applyBitFlipNoise = defaultParameters.applyBitFlipNoise

        MyParameters.bitFlipNoise = defaultParameters.bitFlipNoise

        MyParameters.applyPhaseDumingNoise = defaultParameters.applyPhaseDumingNoise

        MyParameters.phaseDumingNoise = defaultParameters.phaseDumingNoise

        MyParameters.doOneKfold = defaultParameters.doOneKfold

        MyParameters.doOneScalar = defaultParameters.doOneScalar

        MyParameters.onlyScalarValue = defaultParameters.onlyScalarValue

        MyParameters.usePercentageOfData = defaultParameters.usePercentageOfData

        MyParameters.PercentageOfData = defaultParameters.PercentageOfData

        MyParameters.savingFileName = defaultParameters.savingFileName

        # MyParameters.savedModelsFolder = 'saved_models'

        MyParameters.savedModelsFolder = defaultParameters.savedModelsFolder
        


