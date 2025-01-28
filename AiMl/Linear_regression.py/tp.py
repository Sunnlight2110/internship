import matplotlib.pyplot as plt
import numpy as np

# Define the machine learning models and their attributes for the spider web chart
models = [
    "Linear Regression", "Logistic Regression", "Decision Trees", "Random Forests", 
    "SVM", "KNN", "Naive Bayes", "K-Means", "Hierarchical Clustering", 
    "PCA", "Neural Networks", "Deep Learning", "Gradient Boosting", "AdaBoost", "Reinforcement Learning"
]

# Define attributes for each model: 0 = Simple, 1 = Complex, 2 = Medium (for simplicity)
complexity = [
    0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 2, 2, 1, 1, 2
]

use_case = [
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2
]

scalability = [
    1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2
]

accuracy = [
    1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2
]

# Number of variables
num_vars = len(models)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Make the plot a circle by repeating the first value at the end
complexity += complexity[:1]
use_case += use_case[:1]
scalability += scalability[:1]
accuracy += accuracy[:1]
angles += angles[:1]

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot data
ax.plot(angles, complexity, linewidth=2, linestyle='solid', label='Complexity')
ax.plot(angles, use_case, linewidth=2, linestyle='solid', label='Use Case')
ax.plot(angles, scalability, linewidth=2, linestyle='solid', label='Scalability')
ax.plot(angles, accuracy, linewidth=2, linestyle='solid', label='Accuracy')

# Fill the area under each plot
ax.fill(angles, complexity, alpha=0.25)
ax.fill(angles, use_case, alpha=0.25)
ax.fill(angles, scalability, alpha=0.25)
ax.fill(angles, accuracy, alpha=0.25)

# Set the labels for each model
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(models, fontsize=10, ha='right')

# Add title
plt.title("Machine Learning Models Comparison (Spider Web)", size=15, y=1.1)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Show the plot
plt.show()
