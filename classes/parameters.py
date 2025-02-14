import numpy as np

class MyParameters:

    # 0 = wine data
    dataType = 0

    #0 = amplitude embedding, 1 = angle embedding, 2 = custom embedding
    featureMapType = 0

    pca_components = 13

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


    allDataTypes=['Wine Data']

    featureMaps=['Amplitude Embedding', 'Angle Embedding', 'Custom Embedding']