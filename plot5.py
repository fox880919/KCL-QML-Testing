import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from data.my_dataframe import MyDataFrame

from plot_data_class import MyPlotData

myPlotData = MyPlotData()

allModels, allModelsLabels = myPlotData.getAllModelsData2()

# print(f'len(allModels): {len(allModels)}')

# print(f'len(allModelsLabels): {len(allModelsLabels)}')

# for index, model in enumerate(allModels):
#     print(f'len(allModels[{index}]): {len(allModels[index])}')

#     print(f'len(allModelsLabels[{index}]): {len(allModelsLabels[index])}')



# print(f'allModels[1]: {allModels[1]}')

# print(f'allModelsLabels[1]: {allModelsLabels[1]}')

# print(f'allModels: {allModels}')

# print(f'allModelsLabels: {allModelsLabels}')

fig = plt.figure(figsize=(10, 8))

modelsLabels = []

samples = np.arange(1, 17)  # Sample numbers (1 to 10)

# Create a 3D plot
fig = plt.figure(figsize=(40, 12))
ax = fig.add_subplot(111, projection='3d')

for index, model in enumerate(allModelsLabels):
    # modelsLabels.append(f'model_{index}')
    modelsLabels.append(f'M#{index}')

# print(f'modelsLabels: {modelsLabels}')

for i, model in enumerate(modelsLabels):

    # if i == 0:
        y = np.full_like(samples, i)  # Assign a constant y-value for each model

        ax.scatter(samples, y, allModels[i], label=modelsLabels[i])


# Add labels and title
ax.set_xlabel('KFolds')
ax.set_ylabel('Models')
ax.set_zlabel('Accuracy')
ax.set_xticks(np.arange(len(allModels[0])))  # Set y-ticks to model identifiers
ax.set_yticks(np.arange(len(allModels)))  # Set y-ticks to model identifiers
ax.set_yticklabels(modelsLabels)  # Label y-ticks with model names
ax.set_title('Model Accuracies in 3D Space')

# Add a legend
ax.legend()

# output_file = 'models_accuracies.html'  # You can change the file name and extension
# plt.savefig(output_file, dpi=300, bbox_inches='tight')

# Show plot
plt.show()



# # # # Sample data
# # # samples = np.arange(1, 11)  # Sample numbers (1 to 10)
# # # models = ['Model 1', 'Model 2', 'Model 3']  # Model identifiers
# # # accuracy_model1 = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94]  # Model 1 accuracies
# # # accuracy_model2 = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94]  # Model 2 accuracies
# # # accuracy_model3 = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94]  # Model 3 accuracies

# # # # Create a 3D plot
# # # fig = plt.figure(figsize=(10, 8))
# # # ax = fig.add_subplot(111, projection='3d')

# # # # Plot each model's accuracies
# # # for i, model in enumerate(models):
# # #     y = np.full_like(samples, i)  # Assign a constant y-value for each model
# # #     if model == 'Model 1':
# # #         ax.scatter(samples, y, accuracy_model1, label=model, marker='o')
# # #     elif model == 'Model 2':
# # #         ax.scatter(samples, y, accuracy_model2, label=model, marker='s')
# # #     elif model == 'Model 3':
# # #         ax.scatter(samples, y, accuracy_model3, label=model, marker='^')

# # # # Add labels and title
# # # ax.set_xlabel('Samples')
# # # ax.set_ylabel('Models')
# # # ax.set_zlabel('Accuracy')
# # # ax.set_yticks(np.arange(len(models)))  # Set y-ticks to model identifiers
# # # ax.set_yticklabels(models)  # Label y-ticks with model names
# # # ax.set_title('Model Accuracies in 3D Space')

# # # # Add a legend
# # # ax.legend()

# # # # Show plot
# # # plt.show()