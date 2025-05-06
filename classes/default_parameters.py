import numpy as np

class DefaultParameters:
    
    
    #not used anymore
    useParametersClassParameters = True

    roundNumber = 1

    inputNumber = 1

    showProgressDetails = True

    n_folds = 15
    # 0 = wine data 1 = load digits 2 = credit card 3 = mnist 4= custom
    # dataType = 0
    
    dataType = 0

    #0 = amplitude embedding, 1 = angle embedding, 2 = custom embedding
    featureMapType = 0

    amplitudeNQubits = 10
    phasenqubits = 5

    alwaysUsePCA = False 

    # pca_components = 8
    pca_components = 8

    applyMRs = True

    AskUserToApplyMRs = False

    askUserToInputParameters = False

    applyScalarValue = False
    scaleValue = 3

    fromScaleValue = 11

    # toScaleValue = 20
    toScaleValue = 12
    
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
    
    allDataTypes=['Wine Data', 'Load Digits', 'Credit Card', 'MNIST', 'Make Classification']

    featureMaps=['Amplitude Embedding', 'Angle Embedding', 'Custom Embedding']

    # savingFileName = 'my_dataframe.csv'

    # savingFileName = 'my_dataframe_noisy1.csv'

    savingFileName = 'my_dataframe_all_extra_extra_noise_data11.csv'

    # savedModelsFolder = 'saved_models'

    # savedModelsFolder = 'saved_models_noisy1'

    #first naming
    # savedModelsFolder = 'saved_models_all_extra_extra_noise_data11'

    #second naming
    savedModelsFolder = 'saved_models_all/noise1_95_2_95_3_95_4_95_data_1'

    useQiskit = False

    applyDepolarizingChannelNoise = False

    depolarizingChannelNoise = 0.5

    applyAfterEnganglementNoise = False

    afterEnganglementNoise = 0.5

    applyBitFlipNoise = False

    bitFlipNoise = 0.5

    applyAmplitudeDampingNoise = False
    
    amplitudeDampingNoise = 0.5

    applyPhaseDampingNoise = False

    phaseDampingNoise = 0.5

    doOneKfold = False
    
    onlyKFoldValue = 0

    doOneScalar = False

    onlyScalarValue = 3

    usePercentageOfData = False

    PercentageOfData = 0.1
    
    def getModelName(mrNumber, mrValue, fold_index, n_folds):

        return 'SVM'+ str(0) + str(mrNumber)+ '-' + str(mrValue) + '-' + str(fold_index) + '-of-' + str(n_folds)

    def getFullPathModelName(modelName):

        return f'saved_models/{modelName}'


   