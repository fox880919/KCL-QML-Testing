import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import os
import math

# Function to rotate points by a given angle (in degrees)
def rotate_points(points, angle_degrees):
    """
    Rotate 2D points by a given angle (in degrees).
    """
    angle_radians = math.radians(angle_degrees)  # Convert degrees to radians
    rotation_matrix = np.array([
        [math.cos(angle_radians), -math.sin(angle_radians)],
        [math.sin(angle_radians), math.cos(angle_radians)]
    ])
    return np.dot(points, rotation_matrix.T)  # Apply rotation

# Get the current working directory and parent directory
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)  # Go up one level
print('parent_dir:', parent_dir)

# Define the output path for saving the plot
filename = 'plots/mr_2.png'
output_path = os.path.join(parent_dir, filename)
print('output_path:', output_path)

# Step 1: Generate synthetic data
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# Step 2: Rotate the input data by a specified angle (e.g., 45 degrees)
angle_degrees = 45  # Change this to the desired rotation angle
X_rotated = rotate_points(X, angle_degrees)

# Step 3: Train the SVM model on the rotated data
clf = svm.SVC(kernel='linear', C=1000)  # Use a linear kernel
clf.fit(X_rotated, y)

# Step 4: Create a meshgrid to plot the decision boundary and margins
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Step 5: Plot the decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# Step 6: Plot the support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

# Step 7: Plot the rotated data points
ax.scatter(X_rotated[:, 0], X_rotated[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', label='Rotated Data Points')

# Step 8: Add labels and legend
ax.set_xlabel('Feature 1 (x1)')
ax.set_ylabel('Feature 2 (x2)')
ax.set_title(f'SVM Decision Boundary with Margins (Rotated by {angle_degrees} Degrees)')
ax.legend()

# Step 9: Save the plot to a file
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save as PNG

# Step 10: Show the plot (optional)
plt.show()