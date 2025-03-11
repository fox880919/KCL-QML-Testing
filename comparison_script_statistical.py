import sys


sys.path.insert(0, './data')

from data.my_dataframe import MyDataFrame

from classes.statistical_testing import MyStatisticalTesting
import math


myDataFrame = MyDataFrame()


def getAllScores():
    
    nfoldIndex = 16

    totalNotEqual = 0

    originalAccuracyScore = 0
    allOriginalAccuracyScores = []
    allOneMrAccuracyScores = [] 
    allOneKFoldMrsAccuracyScores= []
    AllKFoldMrsAccuracyScores = []

    mrsAccuracyScores = 0

    #1- loop kfold
    for kfoldIndex in range(0, nfoldIndex):
       
        allOneKFoldMrsAccuracyScores = []
        # print('for kfoldIndex: ', kfoldIndex)
        
        originalAccuracyScore = myDataFrame.getModelScoreValue(0, 0, kfoldIndex, nfoldIndex)

        # print(f'originalAccuracyScore for {kfoldIndex}: {originalAccuracyScore}')
        allOriginalAccuracyScores.append(originalAccuracyScore)

        total = 0

        #2- loop mrs
        for mrNumber in range(1, 6):
        # for mrNumber in range(1, 5):
            allOneMrAccuracyScores = []
            # print('for mr: ', mrNumber)

            if mrNumber == 1:
                
                for mrValue in range(2, 20):

                    temp = myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
                    mrsAccuracyScores = mrsAccuracyScores + temp

                    total = total + 1

                    allOneMrAccuracyScores.append(temp)


                    if mrValue == 2:
                        doNothing = True
                        # print(f'mrsAccuracyScores of MrValue {mrValue} for {kfoldIndex}: {temp}')


            # if mrNumber == 2:

            #     for mrValue in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    
            #         mrsAccuracyScores = mrsAccuracyScores + myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
             
            #         total = total + 1

            # if mrNumber == 3:
            
            #     mrValue = 0
            #     mrsAccuracyScores = mrsAccuracyScores + myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
            
            #     total = total + 1
 
            # if mrNumber == 4:
                
            #     mrValue = 0                
            #     mrsAccuracyScores = mrsAccuracyScores + myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
            
            #     total = total + 1
            
            # if mrNumber == 5:

            #     for mrValue in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:

            #         mrsAccuracyScores = mrsAccuracyScores + myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
                
            #         total = total + 1 

            allOneKFoldMrsAccuracyScores.append(allOneMrAccuracyScores)
        
        AllKFoldMrsAccuracyScores.append(allOneKFoldMrsAccuracyScores)
                   
        # print('for kfoldIndex: ', kfoldIndex)
        # print('mrsAccuracyScores: ', mrsAccuracyScores)
        # print('total: ', total)

        mrsAverageAccuracyScores = mrsAccuracyScores / total

        # print('originalAccuracyScore: ', originalAccuracyScore)

        # print('mrsAverageAccuracyScores: ', mrsAverageAccuracyScores)

        # if originalAccuracyScore == mrsAverageAccuracyScores:
        if math.isclose(originalAccuracyScore, mrsAverageAccuracyScores, rel_tol=1e-07):
            
            doNothing = True
            # print('equal')
        else:

            doNothing = True

            # print('for mr: ', mrNumber)

            # print('originalAccuracyScore: ', originalAccuracyScore)

            # print('mrsAverageAccuracyScores: ', mrsAverageAccuracyScores)

            # print('not equal')

            totalNotEqual = totalNotEqual + 1
    
    
    # print('totalNotEqual: ', totalNotEqual)

    return allOriginalAccuracyScores, AllKFoldMrsAccuracyScores

def getOneMRValueAllKFoldScores(AllKFoldMrsAccuracyScores, MRNumber, MrValueIndex):
    
    oneMRValueAllKFoldScores = []
    for index, value in enumerate(AllKFoldMrsAccuracyScores):

        oneMRValueAllKFoldScores.append(AllKFoldMrsAccuracyScores[index][MRNumber-1][MrValueIndex])

    return oneMRValueAllKFoldScores


def compareResults(allOriginalAccuracyScores, oneMRValueAllKFoldScores):

    
   t_statistic, p_value = MyStatisticalTesting.getPairedTest(allOriginalAccuracyScores, oneMRValueAllKFoldScores)

   return t_statistic, p_value
 

allOriginalAccuracyScores, AllKFoldMrsAccuracyScores = getAllScores()


# print(f'allOriginalAccuracyScores: {allOriginalAccuracyScores}')
# print(f'AllKFoldMrsAccuracyScores: {AllKFoldMrsAccuracyScores[5][0][0]}')


oneMRValueAllKFoldScores = []

# oneMRValueAllKFoldScores = getOneMRValueAllKFoldScores(AllKFoldMrsAccuracyScores, 1, 7)

    # print(f'AllKFoldMrsAccuracyScores: {oneMRValueAllKFoldScores}')

    # print(f'len(AllKFoldMrsAccuracyScores): {len(AllKFoldMrsAccuracyScores[0][0])}')

    # print(f'len(AllKFoldMrsAccuracyScores[0][0]): {AllKFoldMrsAccuracyScores[0][0]}')

    # print(f'len(AllKFoldMrsAccuracyScores[0][1]): {AllKFoldMrsAccuracyScores[1][0]}')

    # print(f'len(AllKFoldMrsAccuracyScores[0][2]): {AllKFoldMrsAccuracyScores[2][0]}')

    # print(f'len(AllKFoldMrsAccuracyScores[0][3]): {AllKFoldMrsAccuracyScores[3][0]}')

    # print(f'len(AllKFoldMrsAccuracyScores[0][4]): {AllKFoldMrsAccuracyScores[4][0]}')

    # print(f'len(AllKFoldMrsAccuracyScores[0][5]): {AllKFoldMrsAccuracyScores[5][0]}')

    # print(f'len(AllKFoldMrsAccuracyScores[0][6]): {AllKFoldMrsAccuracyScores[6][0]}')

    # print(f'len(AllKFoldMrsAccuracyScores[0][7]): {AllKFoldMrsAccuracyScores[7][0]}')

    # print(f'len(AllKFoldMrsAccuracyScores[0][8]): {AllKFoldMrsAccuracyScores[8][0]}')


    # print(f'len(AllKFoldMrsAccuracyScores): {AllKFoldMrsAccuracyScores[0][0][2]}')


    # oneMRValueAllKFoldScores = getOneMRValueAllKFoldScores(AllKFoldMrsAccuracyScores, 1)


# print(f'len(allOriginalAccuracyScores): {len(allOriginalAccuracyScores)}')
# print(f'len(AllKFoldMrsAccuracyScores[0][0]): {len(AllKFoldMrsAccuracyScores[0][0])}')

for index, value in enumerate(AllKFoldMrsAccuracyScores[0][0]):

    # print(f'index: {index}')
    oneMRValueAllKFoldScores = getOneMRValueAllKFoldScores(AllKFoldMrsAccuracyScores, 1, index)

    # print(f'len(oneMRValueAllKFoldScores): {len(oneMRValueAllKFoldScores)}')

    t_statistic, p_value = compareResults(allOriginalAccuracyScores, oneMRValueAllKFoldScores)

#     # # print(f'allOriginalAccuracyScores: {allOriginalAccuracyScores}')

#     # # print(f'AllKFoldMrsAccuracyScores: {oneMRValueAllKFoldScores}')

    print(f'comparison with model#{index+1}, statistic= {t_statistic}, and p-value= {p_value}')
#     # print(f"T-statistic: {t_statistic}")
#     # print(f"P-value: {p_value}")


