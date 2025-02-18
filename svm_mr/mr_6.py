import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import os
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Get the current working directory and parent directory
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)  # Go up one level
print('parent_dir:', parent_dir)

# Define the output path for saving the plot
filename = 'plots/mr_6.png'
output_path = os.path.join(parent_dir, filename)
print('output_path:', output_path)

# Step 1: Generate synthetic data
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# Step 2: Add an extra useless feature
extra_feature = np.random.rand(X.shape[0], 1)  # Random values for the extra feature
X_with_extra = np.hstack((X, extra_feature))  # Append the extra feature to X

# Step 3: Train the SVM model on the modified data
clf = svm.SVC(kernel='linear', C=1000)  # Use a linear kernel
clf.fit(X_with_extra, y)

# Step 4: Create a 3D meshgrid to plot the decision boundary
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

# Define the range for the meshgrid
xlim = (X_with_extra[:, 0].min() - 1, X_with_extra[:, 0].max() + 1)
ylim = (X_with_extra[:, 1].min() - 1, X_with_extra[:, 1].max() + 1)
zlim = (X_with_extra[:, 2].min() - 1, X_with_extra[:, 2].max() + 1)

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30),
                    np.linspace(ylim[0], ylim[1], 30))

# Evaluate the decision function on the meshgrid
zz = (-clf.intercept_[0] - clf.coef_[0][0] * xx - clf.coef_[0][1] * yy) / clf.coef_[0][2]

# Step 5: Plot the decision boundary
ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.5, label='Decision Boundary')

# Step 6: Plot the data points
ax.scatter(X_with_extra[:, 0], X_with_extra[:, 1], X_with_extra[:, 2], c=y, cmap=plt.cm.Paired, edgecolors='k', label='Data Points')

# Step 7: Add labels and legend
ax.set_xlabel('Feature 1 (x1)')
ax.set_ylabel('Feature 2 (x2)')
ax.set_zlabel('Feature 3 (x3)')
ax.set_title('SVM Decision Boundary with Margins (3D)')

# Step 8: Save the plot to a file
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save as PNG

# Step 9: Show the plot (optional)
plt.show()