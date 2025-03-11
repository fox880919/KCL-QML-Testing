import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Example data
metamorphic_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Metamorphic relation values
class1_accuracies = [0.85, 0.87, 0.88, 0.89, 0.90]  # Class 1 accuracies
class2_accuracies = [0.83, 0.86, 0.87, 0.88, 0.89]  # Class 2 accuracies
class3_accuracies = [0.82, 0.84, 0.85, 0.86, 0.88]  # Class 3 accuracies

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each class as a separate line
ax.plot(metamorphic_values, [1] * len(metamorphic_values), class1_accuracies, marker='o', label='Class 1')
ax.plot(metamorphic_values, [2] * len(metamorphic_values), class2_accuracies, marker='s', label='Class 2')
ax.plot(metamorphic_values, [3] * len(metamorphic_values), class3_accuracies, marker='d', label='Class 3')

# Set labels and title
ax.set_xlabel('Metamorphic Relation Values')
ax.set_ylabel('Class')
ax.set_zlabel('Accuracy')
ax.set_title('3D Line Plot: Accuracy Across Metamorphic Relation Values and Classes')

# Set y-axis ticks to show class labels
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Class 1', 'Class 2', 'Class 3'])

# Add legend
ax.legend()

# Show plot
plt.show()