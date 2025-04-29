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


if MyParameters.AskUserToApplyMRs:

    useMetamorphicRelations = myMain.DoUserWantToUseMRs()


def loopThroughParametersForMRs(mrNumber, value):
    
    roundMessage = 'starting round of '+ str(mrNumber) + ' and value ' + str(value)
    print(roundMessage)

    myMain.useMR(mrNumber, value)



def start():
    for i in range(0, 6):

        if i == 0:

            loopThroughParametersForMRs(i, 0)

            # MyParameters.featureMapType = 1
            # loopThroughParametersForMRs(0, 0)  

            # MyParameters.featureMapType = 2
            # loopThroughParametersForMRs(0, 0)  

        if i == 1:

            for j in range(MyParameters.fromScaleValue, MyParameters.toScaleValue):
            # for j in range(4, 20):

                loopThroughParametersForMRs(i, j)

        # if i == 2:

        #     for j in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        #         loopThroughParametersForMRs(i, j)

        # if i == 3:

        #     loopThroughParametersForMRs(i, 0)

        # if i == 4:

        #     loopThroughParametersForMRs(i, 0)

        # if i == 5:

        #     for j in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:

        #         loopThroughParametersForMRs(i, j)


# print(f'savingModelFolderName: {MyParameters.getSavingModelFolderName()}')

# modelName = 'svm00'
# print(f'getSavingModelFolder: {MyParameters.getFullPathModelName(modelName)}')

# start()

