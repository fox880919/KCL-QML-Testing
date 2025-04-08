import matplotlib.pyplot as plt
import numpy as np
from data.my_dataframe import MyDataFrame



myDataFrame = MyDataFrame()
nfoldIndex = 16


# Example data
metamorphic_values = range(1, 20)  # Metamorphic relation values


allOriginalKFoldModels = []
allOriginalKFoldModelsLabels = []

for kfoldIndex in range(0, nfoldIndex):
        mrsAccuracyScores = myDataFrame.getModelScoreValue(0, 0, kfoldIndex, nfoldIndex)
        allOriginalKFoldModels.append(mrsAccuracyScores)
        allOriginalKFoldModelsLabels.append(f'SVM00-{0}-{kfoldIndex+1}-of-{nfoldIndex}')

print('allOriginalKFoldModels: ', allOriginalKFoldModels)
print('allOriginalKFoldModelsLabels: ', allOriginalKFoldModelsLabels)

allKFoldModels = []
allRangeModels = []

temp = []

allKFoldModelsLabels = []
allRangeModelsLabels = []

mrsAccuracyScores = 0

for kfoldIndex in range(0, nfoldIndex):

    allRangeModels.append(allOriginalKFoldModels[kfoldIndex])
    allRangeModelsLabels.append(allOriginalKFoldModelsLabels[kfoldIndex])

    temp = []

    mrNumber = 1
                    
    for mrValue in range(2, 20):
        
        mrsAccuracyScores = myDataFrame.getModelScoreValue(mrNumber, mrValue, kfoldIndex, nfoldIndex)
        
        allRangeModels.append(mrsAccuracyScores)
        allRangeModelsLabels.append(f'SVM01-{mrValue}-{kfoldIndex+1}-of-{nfoldIndex}')

    allKFoldModels.append(allRangeModels)
    allKFoldModelsLabels.append(allRangeModelsLabels)

    allRangeModels = []
    allRangeModelsLabels=[]

print('len(allKFoldModels)', len(allKFoldModels))

print('len(allKFoldModelsLabels)', len(allKFoldModelsLabels))

print('len(allKFoldModels[0])', len(allKFoldModels[0]))

print('len(allKFoldModelsLabels[0])', len(allKFoldModelsLabels[0]))


print('allKFoldModels', allKFoldModels[0])

print('allKFoldModelsLabels', allKFoldModelsLabels[0])

plt.figure(figsize=(10, 6))

for firstIndex in range(len(allKFoldModels)):

    tempAllRangeModels = allKFoldModels[firstIndex]
    tempAllRangeModelsLabels = allKFoldModelsLabels[firstIndex]

    # for secondInndex in range(len(tempAllRangeModels)):
        
    #     # tempModelAccuracy = tempAllRangeModels[secondInndex] 
    #     # tempModelLabel = tempAllRangeModelsLabels[secondInndex] 

    plt.plot(metamorphic_values, tempAllRangeModels, marker='o', label=f'nfold-{firstIndex}')

plt.xlabel('Scaling Values')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Across scaling Values and nfold')
plt.legend()
plt.grid(True)
plt.show()
