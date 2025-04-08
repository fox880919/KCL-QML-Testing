
from data.my_dataframe import MyDataFrame

from classes.parameters import MyParameters

class MyPlotData:

    myDataFrame = MyDataFrame()
    nfoldIndex = 16


    def getOriginalModelsData():
        # Example data
        metamorphic_values = range(1, 20)  # Metamorphic relation values


        allOriginalKFoldModels = []
        allOriginalKFoldModelsLabels = []

        for kfoldIndex in range(0, MyPlotData.nfoldIndex):
                mrsAccuracyScores = MyPlotData.myDataFrame.getModelScoreValue(0, 0, kfoldIndex, MyPlotData.nfoldIndex)
                allOriginalKFoldModels.append(mrsAccuracyScores)
                allOriginalKFoldModelsLabels.append(f'SVM00-{0}-{kfoldIndex+1}-of-{MyPlotData.nfoldIndex}')

        print('allOriginalKFoldModels: ', allOriginalKFoldModels)
        print('allOriginalKFoldModelsLabels: ', allOriginalKFoldModelsLabels)
        return allOriginalKFoldModels, allOriginalKFoldModelsLabels
       

    def getAllModelsData(self):
         
        allOriginalKFoldModels, allOriginalKFoldModelsLabels = MyPlotData.getOriginalModelsData()
        
        allKFoldModels = []
        allRangeModels = []

        allKFoldModelsLabels = []
        allRangeModelsLabels = []

        mrNumber = 1

        mrsAccuracyScores = 0

        for kfoldIndex in range(0, MyPlotData.nfoldIndex):

            allRangeModels.append(allOriginalKFoldModels[kfoldIndex])
            allRangeModelsLabels.append(allOriginalKFoldModelsLabels[kfoldIndex])
                            
            for mrValue in range(MyParameters.fromScaleValue, MyParameters.toScaleValue):
                
                mrsAccuracyScores = MyPlotData.myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, MyPlotData.nfoldIndex)
                
                allRangeModels.append(mrsAccuracyScores)
                allRangeModelsLabels.append(f'SVM01-{mrValue}-{kfoldIndex+1}-of-{MyPlotData.nfoldIndex}')

            allKFoldModels.append(allRangeModels)
            allKFoldModelsLabels.append(allRangeModelsLabels)
            
            allRangeModels = []
            allRangeModelsLabels=[]

        return allKFoldModels, allKFoldModelsLabels
    
    #returns allvalues[allkfolds[]]
    def getAllModelsData2(self):
        

        allModels = []
        allLabels = []
        
        tempModels = []
        tempLabels = []

        mrNumber = 1

        for kfoldIndex in range(0, MyPlotData.nfoldIndex):

                mrsAccuracyScores = MyPlotData.myDataFrame.getModelScoreValue(0, 0, kfoldIndex, MyPlotData.nfoldIndex)
                
                tempModels.append(mrsAccuracyScores)

                tempLabels.append(f'SVM00-{0}-{kfoldIndex+1}-of-{MyPlotData.nfoldIndex}')

        allModels.append(tempModels)
        allLabels.append(tempLabels)

  
        # print(f'allModels: {allModels}')
        # print(f'allLabels: {allLabels}')
        # return allModels, allLabels



        for mrValue in range(MyParameters.fromScaleValue, MyParameters.toScaleValue):

            tempModels = []
            tempLabels = []   
            # print(f'in mrValue:{mrValue}')
            for kfoldIndex in range(MyPlotData.nfoldIndex):
                    
                # print(f'in kfoldIndex:{kfoldIndex}')

                mrsAccuracyScores = MyPlotData.myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, MyPlotData.nfoldIndex)
                tempModels.append(mrsAccuracyScores)
                tempLabels.append(f'SVM01-{mrValue}-{kfoldIndex+1}-of-{MyPlotData.nfoldIndex}')
        
      
            # print(f'len(allModels): {len(allModels)}')
            # print(f'len(allLabels): {len(allLabels)}')        
            allModels.append(tempModels)
            allLabels.append(tempLabels)
        
        # print(f'len(allModels[0]): {len(allModels[0])}')
        # print(f'len(allLabels[0]): {len(allLabels[0])}')
        return allModels, allLabels

    

            
