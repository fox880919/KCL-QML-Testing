import numpy as np

import pennylane as qml

import pennylane_qiskit

from qiskit_ibm_runtime import QiskitRuntimeService


# from default_parameters import DefaultParameters

from classes.default_parameters import DefaultParameters 

class MyParameters: 

    backend = {}

    time_format = "%Y-%m-%d %H:%M:%S"

    timeBeforeBackend = 0

    timeAfterBackend = 0


    ####
    useIBMBackEndService = DefaultParameters.useIBMBackEndService

    justCalculateJobTime = DefaultParameters.justCalculateJobTime

    usePrecomputedKernel = DefaultParameters.usePrecomputedKernel

    useParametersClassParameters = DefaultParameters.useParametersClassParameters

    roundNumber = 1

    inputNumber = 1

    showProgressDetails = DefaultParameters.showProgressDetails

    n_folds = DefaultParameters.n_folds
    
    # 0 = wine data 1 = load digits 2 = credit card 3 = mnist 4= custom
    # dataType = 0
    # 2 = creditcard
    # 4 = mnist
    dataType = DefaultParameters.dataType

    #0 = amplitude embedding, 1 = angle embedding, 2 = custom embedding
    featureMapType = DefaultParameters.featureMapType

    amplitudeNQubits = DefaultParameters.amplitudeNQubits
    angleNQubits = DefaultParameters.angleNQubits

    phasenqubits = DefaultParameters.phasenqubits

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

    useQiskit = DefaultParameters.useQiskit

    applyDepolarizingChannelNoise = DefaultParameters.applyDepolarizingChannelNoise

    depolarizingChannelNoise = DefaultParameters.depolarizingChannelNoise

    applyAfterEnganglementNoise = DefaultParameters.applyAfterEnganglementNoise

    afterEnganglementNoise = DefaultParameters.afterEnganglementNoise

    applyBitFlipNoise = DefaultParameters.applyBitFlipNoise

    bitFlipNoise = DefaultParameters.bitFlipNoise

    applyAmplitudeDampingNoise = DefaultParameters.applyAmplitudeDampingNoise

    amplitudeDampingNoise = DefaultParameters.amplitudeDampingNoise

    applyPhaseDampingNoise = DefaultParameters.applyPhaseDampingNoise

    phaseDampingNoise = DefaultParameters.phaseDampingNoise

    doOneKfold = DefaultParameters.doOneKfold

    onlyKFoldValue = DefaultParameters.onlyKFoldValue

    doOneScalar = DefaultParameters.doOneScalar

    onlyScalarValue = DefaultParameters.onlyScalarValue

    usePercentageOfData = DefaultParameters.usePercentageOfData

    PercentageOfData =DefaultParameters.PercentageOfData

    def getModelName(mrNumber, mrValue, fold_index, n_folds):

        return 'SVM'+ str(0) + str(mrNumber)+ '-' + str(mrValue) + '-' + str(fold_index) + '-of-' + str(n_folds)

    # naming 1
    # def getFullPathModelName(modelName):

    #     return f'saved_models/{modelName}'

    def getFullPathModelNamefromOutisde(self, modelName):

        print('inside getFullPathModelNamefromOutisde')
        return MyParameters.getFullPathModelName(modelName)

    #naming 2
    def getFullPathModelName(modelName):

        print('inside getFullPathModelName')

        return f'{MyParameters.getSavingModelFolderName()}/{modelName}'
    
    def isNoiseUsed():

        return True if MyParameters.applyDepolarizingChannelNoise or MyParameters.applyAfterEnganglementNoise or MyParameters.applyBitFlipNoise or MyParameters.applyAmplitudeDampingNoise or MyParameters.applyPhaseDampingNoise else False

    
    def getSavingFileName():

        # return MyParameters.savingFileName

        finalFileName = 'my_data_frame_all'

        return finalFileName
    

        original = 'my_dataframe_'

        # noise0 = 'noise' if MyParameters.applyDepolarizingChannelNoise or MyParameters.applyAfterEnganglementNoise or MyParameters.applyBitFlipNoise or MyParameters.applyPhaseDampingNoise else 'no_noise_'
        noise0 = 'noise' if MyParameters.isNoiseUsed() else 'no_noise_'

        noise1 = f'_1_{MyParameters.depolarizingChannelNoise}_' if MyParameters.applyDepolarizingChannelNoise else ''

        noise2 = f'2_{MyParameters.afterEnganglementNoise}_' if MyParameters.applyAfterEnganglementNoise else ''

        noise3 = f'3_{MyParameters.bitFlipNoise}_' if MyParameters.applyBitFlipNoise else ''

        noise4 = f'4_{MyParameters.amplitudeDampingNoise}_' if MyParameters.applyAmplitudeDampingNoise else ''

        noise5 = f'_5_{MyParameters.phaseDampingNoise}_' if MyParameters.applyPhaseDampingNoise else ''
    
        data = f'data{MyParameters.dataType}'

        savingFileName = f'{original}{noise0}{noise1}{noise2}{noise3}{noise4}{noise5}{data}' 

        return savingFileName
    
    def initiateBackend():
    
        print('calling backend')
        service = QiskitRuntimeService()

        backend = service.backend(
            "ibm_brisbane"
        )
        MyParameters.backend = backend

        print(f'backend is: {MyParameters.backend}')
    
    def getDevice():

        if MyParameters.useIBMBackEndService:

            # print(f'from feature map type: {featureMapType}')
            if MyParameters.backend == {}:
                MyParameters.initiateBackend()
            else:
                print(f'already initiated backend: {MyParameters.backend}')
            # dev = qml.device(MyParameters.getDeviceType(),
            #     backend = MyParameters.backend, wires = 5)

            dev = qml.device(MyParameters.getDeviceType(), wires=MyParameters.amplitudeNQubits, backend=MyParameters.backend, shots=1024)

        else:

            print(f'getDevice() is not use IBM backend')

            dev = qml.device(MyParameters.getDeviceType(), wires=MyParameters.amplitudeNQubits)

            if MyParameters.featureMapType == 1:

             print(f'in getDevice, MyParameters.featureMapType == 1')
             dev = qml.device(MyParameters.getDeviceType(), wires=MyParameters.angleNQubits)

        
        # if MyParameters.useIBMBackEndService == True:

            # dev = qml.device(MyParameters.getDeviceType(), wires=MyParameters.amplitudeNQubits, backend="ibmq_qasm_simulator", shots=1024)
            # dev = qml.device(MyParameters.getDeviceType(), wires=MyParameters.amplitudeNQubits, backend='brisbane', shots=1024)

        # print(f'in MyParamaeters.getDevice with MyParameters.useIBMBackEndService = {MyParameters.useIBMBackEndService}')

        print(f'in MyParamaeters.getDevice with dev = {dev}')

        return dev
    

    
    def getDeviceType():

        deviceType = 'lightning.qubit'

        if MyParameters.useIBMBackEndService == True:

            # deviceType = "qiskit.ibmq"
            # deviceType = "qiskit.ibm"
            deviceType = "qiskit.remote"

        else:    
            deviceType = 'default.mixed' if MyParameters.applyDepolarizingChannelNoise or MyParameters.applyAfterEnganglementNoise or MyParameters.applyBitFlipNoise or MyParameters.applyPhaseDampingNoise else 'lightning.qubit'

        print(f'device type: {deviceType}')
        
        return deviceType
    
    def getSavingModelFolderName():

        # return 'saved_models_all/noise1_95_2_95_3_95_4_95_data_1'

        qiskit = 'qiskit_' if MyParameters.useQiskit else ''

        noise0 = 'noise' if MyParameters.applyDepolarizingChannelNoise or MyParameters.applyAfterEnganglementNoise or MyParameters.applyBitFlipNoise or MyParameters.applyPhaseDampingNoise else 'no_noise_'

        noise1 = f'_1_{MyParameters.depolarizingChannelNoise}_' if MyParameters.applyDepolarizingChannelNoise else ''

        noise2 = f'2_{MyParameters.afterEnganglementNoise}_' if MyParameters.applyAfterEnganglementNoise else ''

        noise3 = f'3_{MyParameters.bitFlipNoise}_' if MyParameters.applyBitFlipNoise else ''

        noise4 = f'4_{MyParameters.amplitudeDampingNoise}_' if MyParameters.applyAmplitudeDampingNoise else ''

        noise5 = f'_5_{MyParameters.phaseDampingNoise}_' if MyParameters.applyPhaseDampingNoise else ''
    
        data = f'data{MyParameters.dataType}'


        return f'saved_models_all/{qiskit}{noise0}{noise1}{noise2}{noise3}{noise4}{noise5}{data}'
    

    def resetParameters():

        
        defaultParameters = DefaultParameters()

        MyParameters.useIBMBackEndService = DefaultParameters.useIBMBackEndService

        MyParameters.justCalculateJobTime = DefaultParameters.justCalculateJobTime
        
        MyParameters.usePrecomputedKernel = DefaultParameters.usePrecomputedKernel

        MyParameters.useParametersClassParameters = defaultParameters.useParametersClassParameters

        MyParameters.roundNumber = defaultParameters.roundNumber + 1

        MyParameters.inputNumber = defaultParameters.inputNumber + 1

        MyParameters.showProgressDetails = defaultParameters.showProgressDetails

        MyParameters.n_folds = defaultParameters.n_folds
        # 0 = wine data
        # MyParameters.dataType = 0
        # 2 = creditcard
        MyParameters.dataType = defaultParameters.dataType


        #0 = amplitude embedding, 1 = angle embedding, 2 = custom embedding
        MyParameters.featureMapType = defaultParameters.featureMapType

        MyParameters.amplitudeNQubits = DefaultParameters.amplitudeNQubits

        MyParameters.angleNQubits = DefaultParameters.angleNQubits

        MyParameters.phasenqubits = DefaultParameters.phasenqubits
        
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



        MyParameters.useQiskit = DefaultParameters.useQiskit

        MyParameters.applyDepolarizingChannelNoise = defaultParameters.applyDepolarizingChannelNoise

        MyParameters.depolarizingChannelNoise = defaultParameters.depolarizingChannelNoise

        MyParameters.applyAfterEnganglementNoise = defaultParameters.applyAfterEnganglementNoise

        MyParameters.afterEnganglementNoise = defaultParameters.afterEnganglementNoise

        MyParameters.applyBitFlipNoise = defaultParameters.applyBitFlipNoise

        MyParameters.bitFlipNoise = defaultParameters.bitFlipNoise

        MyParameters.applyAmplitudeDampingNoise = defaultParameters.applyAmplitudeDampingNoise

        MyParameters.amplitudeDampingNoise = defaultParameters.amplitudeDampingNoise

        MyParameters.applyPhaseDampingNoise = defaultParameters.applyPhaseDampingNoise

        MyParameters.phaseDampingNoise = defaultParameters.phaseDampingNoise

        MyParameters.doOneKfold = defaultParameters.doOneKfold

        MyParameters.doOneScalar = defaultParameters.doOneScalar

        MyParameters.onlyScalarValue = defaultParameters.onlyScalarValue

        MyParameters.usePercentageOfData = defaultParameters.usePercentageOfData

        MyParameters.PercentageOfData = defaultParameters.PercentageOfData

        MyParameters.savingFileName = defaultParameters.savingFileName

        # MyParameters.savedModelsFolder = 'saved_models'

        MyParameters.savedModelsFolder = defaultParameters.savedModelsFolder
        


