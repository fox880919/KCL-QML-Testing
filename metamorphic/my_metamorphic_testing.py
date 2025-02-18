

from qsvm.my_kernel import MyKernel

from my_metamorphic_relations import MyMetamorphicRelations

import numpy as np

from classes.parameters import MyParameters


# from data_manager import DataManager

import numpy as np

class MyMetamorphicTesting:

    qsvm = {}
    myMetamorphicRelations = {}
    type = 1

    def myInit(self):
        
        MyMetamorphicTesting.qsvm = MyKernel()
        MyMetamorphicTesting.myMetamorphicRelations = MyMetamorphicRelations()

    def pickTypeAndPrepareData(self, type):

        MyMetamorphicTesting.type = type

        np, input_tr, input_test, output_tr, output_test = MyMetamorphicTesting.qsvm.pickKernel(type)

        return input_tr, input_test, output_tr, output_test


    def getCiruitDetails(self):

        return MyMetamorphicTesting.qsvm.nqubits, MyMetamorphicTesting.qsvm.kernel_circ

    #1
    def scaleInputData(self, input_tr, input_test, scaleValue):

        # print('input_tr.shape')
        # print(input_tr.shape)

        mr = MyMetamorphicTesting.myMetamorphicRelations

        input_tr, input_test = mr.metamorphic_feature_scaling(input_tr, input_test, scaleValue)

        #dataManager = DataManager()

        #input_tr, input_test = dataManager.implementPCA(input_tr, input_test, 8)

        #MyMetamorphicTesting.qsvm.startSVC(input_tr, input_test, output_tr, output_test)

        return input_tr, input_test

    
      #2
    def rotateInputDataWithAngle(self, x, angle):

        # print('rotating input with angle: ', angle)

        # print('before rotation X')
        # print(x)

        mr = MyMetamorphicTesting.myMetamorphicRelations

        rotatedX = mr.metamorphic_feature_rotation_with_angle(x, angle)

        # print('after rotation rotatedX')
        # print(rotatedX)

        return rotatedX
    
    #3
    def permutateInputData(self, input, output):

        mr = MyMetamorphicTesting.myMetamorphicRelations

        permutatedInput, permutatedOutput = mr.metamorphic_feature_permutation(input, output)

        return permutatedInput, permutatedOutput

    
    #4
    def invertAllLabels(self, y_train, y_test, num_classes):

        mr = MyMetamorphicTesting.myMetamorphicRelations

        # print('1- y_train')
        # print(y_train)
        
        y_train_inverted, y_test_inverted = mr.metamorphic_invert_all_labels_multiclass(y_train, y_test, num_classes)

        return y_train_inverted, y_test_inverted
    
    #5
    def perturbParameters(self, x, delta=0.1):

        mr = MyMetamorphicTesting.myMetamorphicRelations

        xPerturbed = mr.perturb_parameters(x, delta= delta)
        
        return xPerturbed
    
     #--
    def modifyCircuitDepth(self, type):
  
        mr = MyMetamorphicTesting.myMetamorphicRelations

        newCircuit = mr.modify_circuit_depth(type)

        return newCircuit
    
    #6
    def addingAdditionalFeature(self, x_tr, x_test):
        
        mr = MyMetamorphicTesting.myMetamorphicRelations

        new_x_tr, new_x_test =  mr.addingAdditionalFeature(x_tr, x_test)

        MyParameters.pca_components = + MyParameters.pca_components + 1
        return new_x_tr, new_x_test

    #7
    def addingRedundantInputsAndOutputs(self, x_tr, x_test, y_tr, y_test):
        
        mr = MyMetamorphicTesting.myMetamorphicRelations

        new_x_tr, new_x_test, new_y_tr, new_y_test =  mr.addingRedundantInputsAndOutputs(x_tr, x_test, y_tr, y_test)
        return new_x_tr, new_x_test, new_y_tr, new_y_test



    def evaluateData(self, input_tr, input_test, output_tr, output_test):
        
        qsvm = MyMetamorphicTesting.qsvm

        qsvm.startSVC(input_tr, input_test, output_tr, output_test)

    def evaluateDataWithModifiedCircuit(self, input_tr, input_test, output_tr, output_test, circuitDepth, nqubits, modifiedCircuit):
        
        qsvm = MyMetamorphicTesting.qsvm

        qsvm.startSVCWithModifiedCircuit(input_tr, input_test, output_tr, output_test, circuitDepth, nqubits, modifiedCircuit)
