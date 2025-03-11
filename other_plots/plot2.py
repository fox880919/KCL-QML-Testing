import matplotlib.pyplot as plt
import numpy as np
from data.my_dataframe import MyDataFrame

from plot_data_class import MyPlotData

myPlotData = MyPlotData()

metamorphic_values = range(1, 20)  # Metamorphic relation values

allKFoldModels, allKFoldModelsLabels = myPlotData.getAllModelsData()

print('len(allKFoldModels)', len(allKFoldModels))

print('len(allKFoldModelsLabels)', len(allKFoldModelsLabels))

print('len(allKFoldModels[0])', len(allKFoldModels[0]))

print('len(allKFoldModelsLabels[0])', len(allKFoldModelsLabels[0]))


print('allKFoldModels', allKFoldModels[0])

print('allKFoldModelsLabels', allKFoldModelsLabels[0])



plt.figure(figsize=(10, 6))

width = 0.1  # the width of the bars


x = np.arange(len(allKFoldModelsLabels))  # the label locations

for index in range(len(x)):

    z = np.arange(len(allKFoldModelsLabels[index]))  # the label locations

    # print(f'len(allKFoldModels): {len(allKFoldModels)}')
    # print(f'len(allKFoldModels[{indexndex}]): {len(allKFoldModels[indexndex])}')

    tempAllRangeModels = allKFoldModels[index]
    tempAllRangeModelsLabels = allKFoldModelsLabels[index]

    # for secondInndex in range(len(tempAllRangeModels)):
        
    #     # tempModelAccuracy = tempAllRangeModels[secondInndex] 
    #     # tempModelLabel = tempAllRangeModelsLabels[secondInndex] 

    # if index == 0:
    myLabel = f'Model {index+1}'
    plt.bar(z - index * width/16, tempAllRangeModels, width, label=myLabel)

plt.xlabel('Folds')
plt.ylabel('Accuracy')
plt.title('n-Fold Cross-Validation Accuracy Comparison')
plt.xticks(x, allKFoldModelsLabels)
plt.legend()
plt.grid(True)
plt.show()


# # Example data
# folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
# model1_accuracies = [0.85, 0.86, 0.87, 0.88, 0.89]  # Model 1 accuracies per fold
# model2_accuracies = [0.83, 0.84, 0.85, 0.86, 0.87]  # Model 2 accuracies per fold

# x = np.arange(len(folds))  # the label locations
# width = 0.35  # the width of the bars

# plt.figure(figsize=(10, 6))
# plt.bar(x - width/2, model1_accuracies, width, label='Model 1')
# plt.bar(x + width/2, model2_accuracies, width, label='Model 2')

# plt.xlabel('Folds')
# plt.ylabel('Accuracy')
# plt.title('n-Fold Cross-Validation Accuracy Comparison')
# plt.xticks(x, folds)
# plt.legend()
# plt.grid(True)
# plt.show()