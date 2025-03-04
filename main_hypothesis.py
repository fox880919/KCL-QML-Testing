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

if MyParameters.askUserToInputParameters:

    MyParameters.useParametersClassParameters = myMain.DoUserWantToUseDefaultParameters()

else:
    MyParameters.useParametersClassParameters = True


if MyParameters.AskUserToApplyMRs:

    useMetamorphicRelations = myMain.DoUserWantToUseMRs()

def choose_y_strategy(x):
    if x == 1:
        return st.integers(min_value=2, max_value = 20)  # y must be positive
   
    elif x == 2:
        return st.floats(min_value=0.1, max_value=0.9)  # y must be negative
    
    elif x == 3:
        return st.none()

    elif x == 4:
        return st.none()
   
    elif x == 5:
        
        min_value = 1
        max_value = 20
        step = 2
        return st.integers(min_value=min_value, max_value=max_value).map(lambda val: min_value + (val // step) * step)
    
    else:
        return st.none()  # y can be any integer



@given(mrNumber = st.integers(min_value=1, max_value=5), value = st.floats().flatmap(lambda mrNumber: choose_y_strategy(mrNumber)))
def loopThroughParametersForMRs(mrNumber, value):
    
    roundMessage = 'starting round of '+ str(mrNumber) + ' and value ' + str(value)
    print(roundMessage)

    myMain.useMR(mrNumber, value)


if __name__ == "__main__":
    loopThroughParametersForMRs()

# tryManyParameters()

# loopThroughParametersForMain()

