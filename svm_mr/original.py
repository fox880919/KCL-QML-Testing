import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


import os

current_dir = os.getcwd()

parent_dir = os.path.dirname(current_dir)  # Go up one level

print('parent_dir:', parent_dir)

# filename = 'plots/mr_1_1.png'

filename = 'plots/mr_1_2.png'

output_path = os.path.join(parent_dir, filename)

# output_path = parent_dir / filename  # Use / to join paths

print('output_path:', output_path)


# Step 1: Generate synthetic data
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# Step 2: Train the SVM model
clf = svm.SVC(kernel='linear', C=1000)  # Use a linear kernel
clf.fit(X, y)


# Step 3: Create a meshgrid to plot the decision boundary and margins
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Step 4: Plot the decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# Step 5: Plot the support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

# Step 6: Plot the data points
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', label='Data Points')

# Step 7: Add labels and legend
ax.set_xlabel('Feature 1 (x1)')
ax.set_ylabel('Feature 2 (x2)')
ax.set_title('SVM Decision Boundary with Margins')
ax.legend()

# Step 8: Save the plot to a file
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save as PNG
# plt.savefig('svm_plot.pdf', bbox_inches='tight')  # Save as PDF
# plt.savefig('svm_plot.svg', bbox_inches='tight')  # Save as SVG

# Step 9: Show the plot (optional)
plt.show()