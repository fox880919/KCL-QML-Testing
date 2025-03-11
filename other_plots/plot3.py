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

# Example data: Label agreement matrix (1 = agreement, 0 = disagreement)
label_agreement = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
])

plt.figure(figsize=(8, 6))
sns.heatmap(label_agreement, annot=True, cmap='Blues', cbar=False)
plt.xlabel('Folds')
plt.ylabel('Models')
plt.title('Testing Data Label Agreement Across Models and Folds')
plt.show()