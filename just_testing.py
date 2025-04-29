import matplotlib.pyplot as plt
import numpy as np

# Data from the image
conditions = ['Control', 'LPF-FBS', 'Control', 'LPF-FBS']
expression = [1.0000, 1.3037, 1.1013, 1.5172]
errors = [0.1, 0.1, 0.05, 0.05]  # approximate error bars
labels = ['SiScr HMGCR', 'SiTET2 HMGCR']
colors = ['#0099FF', '#66CCFF']

# Set up the bar plot
x = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars
bars1 = ax.bar(x[:2], expression[:2], width, yerr=errors[:2], label=labels[0], color=colors[0], capsize=5)
bars2 = ax.bar(x[2:], expression[2:], width, yerr=errors[2:], label=labels[1], color=colors[1], capsize=5)

# Add data labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# Aesthetics
ax.set_ylabel('Fold change in expression')
ax.set_xlabel('Media Type\n(Normal vs Lipid-depleted Media)')
ax.set_title('TET2 Silencing Effect on HMGCR Expression', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, 2)
ax.legend(title='')

# Clean up borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()