import sys


sys.path.insert(0, './data')

from data.my_dataframe import MyDataFrame

import math


myDataFrame = MyDataFrame()


def getAllScores():
    
    nfoldIndex = 16

    totalNotEqual = 0

    for kfoldIndex in range(0, nfoldIndex):

        print('for kfoldIndex: ', kfoldIndex)
        
        originalAccuracyScore = myDataFrame.getModelScoreValue(0, 0, kfoldIndex, nfoldIndex)

        mrsAccuracyScores = 0

        total = 0

        for mrNumber in range(1, 6):
        # for mrNumber in range(1, 5):

            # print('for mr: ', mrNumber)

            if mrNumber == 1:
                
                for mrValue in range(2, 20):
                    
                    mrsAccuracyScores = mrsAccuracyScores + myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
             
                    total = total + 1

            # if mrNumber == 2:

            #     for mrValue in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    
            #         mrsAccuracyScores = mrsAccuracyScores + myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
             
            #         total = total + 1

            if mrNumber == 3:
            
                mrValue = 0
                mrsAccuracyScores = mrsAccuracyScores + myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
            
                total = total + 1
 
            if mrNumber == 4:
                
                mrValue = 0                
                mrsAccuracyScores = mrsAccuracyScores + myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
            
                total = total + 1
            
            # if mrNumber == 5:

            #     for mrValue in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:

            #         mrsAccuracyScores = mrsAccuracyScores + myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
                
            #         total = total + 1                    

        # print('for kfoldIndex: ', kfoldIndex)
        # print('mrsAccuracyScores: ', mrsAccuracyScores)
        # print('total: ', total)


        mrsAverageAccuracyScores = mrsAccuracyScores / total

        print('originalAccuracyScore: ', originalAccuracyScore)

        print('mrsAverageAccuracyScores: ', mrsAverageAccuracyScores)

        # if originalAccuracyScore == mrsAverageAccuracyScores:
        if math.isclose(originalAccuracyScore, mrsAverageAccuracyScores, rel_tol=1e-07):

            print('equal')
        else:

            # print('for mr: ', mrNumber)

            # print('originalAccuracyScore: ', originalAccuracyScore)

            # print('mrsAverageAccuracyScores: ', mrsAverageAccuracyScores)

            print('not equal')

            totalNotEqual = totalNotEqual + 1
    
    print('totalNotEqual: ', totalNotEqual)


getAllScores()


