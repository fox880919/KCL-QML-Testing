import numpy as np
import plotly.graph_objects as go

# Assuming MyDataFrame and MyPlotData are custom classes for data handling
from data.my_dataframe import MyDataFrame
from plot_data_class import MyPlotData

# Initialize MyPlotData and get model data
myPlotData = MyPlotData()
allModels, allModelsLabels = myPlotData.getAllModelsData2()

# Print the number of models and their labels for verification
print(f'allModels: {len(allModels)}')  # Should print 19
print(f'allModelsLabels: {len(allModelsLabels)}')  # Should print 19

# Prepare model labels (e.g., M#0, M#1, ..., M#18)
modelsLabels = [f'M#{i}' for i in range(len(allModelsLabels))]
print(f'modelsLabels: {modelsLabels}')

# Define sample numbers (1 to 16 for k-folds)
samples = np.arange(1, 17)

# Create a Plotly 3D scatter plot
fig = go.Figure()

# Add each model's MR values to the plot
for i, model in enumerate(modelsLabels):
    y = np.full_like(samples, i)  # Assign a constant y-value for each model
    fig.add_trace(go.Scatter3d(
        x=samples,  # K-folds
        y=y,        # Model index
        z=allModels[i],  # MR values
        mode='markers',
        marker=dict(size=5),
        name=model  # Model label
    ))

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title='KFolds',
        yaxis_title='Models',
        zaxis_title='MR Values',
        xaxis=dict(tickvals=np.arange(len(allModels[0]))),  # Set y-ticks to model identifiers
        yaxis=dict(tickvals=np.arange(len(modelsLabels))),  # Set y-ticks to model identifiers
    ),
    title='All Models values in 3D Space',
    margin=dict(l=0, r=0, b=0, t=30),  # Adjust margins
) # Closing parenthesis for update_layout

# Save the plot as an interactive HTML file
output_file = 'models_accuracies.html'
fig.write_html(output_file)

# Optionally, display the plot in a browser
fig.show()