# import sys
# sys.path.insert(0, './data')


# from data.my_dataframe import MyDataFrame

# myDataFrame = MyDataFrame()

# accuracyScore = myDataFrame.getModelScoreValue(0,0,1,16)

# print('accuracyScore: ', accuracyScore)




import sys

from classes.time import MyTimeHelper

sys.path.insert(0, './data')
sys.path.insert(1, './metamorphic')
sys.path.insert(2, './classes')
sys.path.insert(3, './qsvm')
sys.path.insert(4, './metamorphic')

from hypothesis import given, strategies as st


from classes.parameters import MyParameters

from classes.read_user_input import ReadUserInput

from data.data_manager import DataManager

from qsvm.my_kernel import MyKernel

from qsvm.my_qsvm import MyQSVM

from data.my_dataframe import MyDataFrame

from metamorphic.my_metamorphic_testing import MyMetamorphicTesting

from qsvm.my_pca import MyPCA

from main_class import MyMain

myMain = MyMain()

if  MyParameters.askUserToInputParameters:

    print(' in  askUserToInputParameters')
    MyParameters.useParametersClassParameters = myMain.DoUserWantToUseDefaultParameters()

else:
    MyParameters.useParametersClassParameters = True


def loopThroughParametersForMRs(mrNumber, value):


    myMain.useMR(mrNumber, value)


i = 2
j = 0.1

loopThroughParametersForMRs(i, j)

