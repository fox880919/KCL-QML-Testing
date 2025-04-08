import numpy as np

class DefaultParameters:

    useParametersClassParameters = True

    n_folds = 16
    # 0 = wine data
    # dataType = 0
    
    # 2 = mnist
    dataType = 0

    #0 = amplitude embedding, 1 = angle embedding, 2 = custom embedding
    featureMapType = 0

    alwaysUsePCA = False 

    # pca_components = 8
    pca_components = 8

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

    # savingFileName = 'my_dataframe_noisy1.csv'

    savingFileName = 'my_dataframe_all_noise_data1.csv'

    # savedModelsFolder = 'saved_models'

    # savedModelsFolder = 'saved_models_noisy1'

    savedModelsFolder = 'saved_models_all_noise_data1'

    applyDepolarizingChannelNoise = True

    depolarizingChannelNoise = 0.2

    applyAfterEnganglementNoise = False

    afterEnganglementNoise = 0.2

    applyBitFlipNoise = False

    bitFlipNoise = 0.2

    applyPhaseDumingNoise = False

    phaseDumingNoise = 0.2

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


   