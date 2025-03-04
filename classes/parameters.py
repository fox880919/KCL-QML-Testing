import numpy as np

class MyParameters:

    useParametersClassParameters = True

    n_folds = 16
    # 0 = wine data
    dataType = 0

    #0 = amplitude embedding, 1 = angle embedding, 2 = custom embedding
    featureMapType = 0

    pca_components = 8

    applyMRs = True

    AskUserToApplyMRs = False

    askUserToInputParameters = False

    applyScalarValue = False
    scaleValue = 3
    
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

    modelName = 'svm00'
    
    allDataTypes=['Wine Data']

    featureMaps=['Amplitude Embedding', 'Angle Embedding', 'Custom Embedding']




    def resetParameters():

        MyParameters.useParametersClassParameters = True

        MyParameters.n_folds = 16
        # 0 = wine data
        MyParameters.dataType = 0

        #0 = amplitude embedding, 1 = angle embedding, 2 = custom embedding
        MyParameters.featureMapType = 0

        MyParameters.pca_components = 8

        MyParameters.applyMRs = True

        MyParameters.AskUserToApplyMRs = False

        MyParameters.askUserToInputParameters = False

        MyParameters.applyScalarValue = False
        MyParameters.scaleValue = 3
        
        MyParameters.applyAngleRotation = False
        MyParameters.angle = 2* np.pi 

        MyParameters.applyPermutation = False

        MyParameters.invertAllLabels = False
        MyParameters.numberOfLabelsClasses = 3

        MyParameters.applyPerturbNoise = False
        MyParameters.perturbNoise = 0.1

        MyParameters.circuitDepth = 2
        MyParameters.applyCircuitDepth = False

        MyParameters.modifyCircuitDepth = False

        MyParameters.addAdditionalFeature = False

        MyParameters.addAdditionalInputsAndOutputs = False

        MyParameters.useTrainedModel= False

        MyParameters.modelName = 'svm00'
        
        MyParameters.allDataTypes=['Wine Data']

        MyParameters.featureMaps=['Amplitude Embedding', 'Angle Embedding', 'Custom Embedding']