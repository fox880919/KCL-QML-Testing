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

for index in range(len(allKFoldModels)):


    # print(f'len(allKFoldModels): {len(allKFoldModels)}')
    # print(f'len(allKFoldModels[{indexndex}]): {len(allKFoldModels[indexndex])}')

    tempAllRangeModels = allKFoldModels[index]
    tempAllRangeModelsLabels = allKFoldModelsLabels[index]

    # for secondInndex in range(len(tempAllRangeModels)):
        
    #     # tempModelAccuracy = tempAllRangeModels[secondInndex] 
    #     # tempModelLabel = tempAllRangeModelsLabels[secondInndex] 

    plt.plot(metamorphic_values, tempAllRangeModels, marker='o', label=f'nfold-{index}')

plt.xlabel('Scaling Values')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Across scaling Values and nfold')
plt.legend()
plt.grid(True)
plt.show()
