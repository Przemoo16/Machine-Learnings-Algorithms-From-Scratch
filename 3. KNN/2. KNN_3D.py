from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# MANUAL #

# Seed
np.random.seed(101)

# HyperParameters
num_features = 3
n_class = 2
num_points = 300

# Generate data
x, y_true = make_classification(n_samples=num_points, n_features=num_features,
                                n_redundant=0, n_informative=2, n_clusters_per_class=n_class)

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y_true, test_size=0.3, random_state=101)

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot training data
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train)
# Title
ax.set_title('Training data')
# Axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')


def distances_func(points, pred_point, labels, k):
    """Compute distance."""
    # Distances between all points (dist, index, label)
    distances = [[np.linalg.norm(point-pred_point), index, label]
                 for index, point, label in zip(range(len(labels)), points, labels)]
    # Sort descending and choose only k values
    return np.array(sorted(distances)[:k])


# List to grab final labels
final_labels = []
# Choose k number
k = 21

# Train and test model
for test_sample in X_test:
    # Distance between each test and training point
    distances = distances_func(X_train, test_sample, y_train, k)
    # Count number of points in each class
    unique, counts = np.unique(distances[:, -1], return_counts=True)
    # Choose class with more points around
    final_labels.append(unique[np.argmax(counts)])
# Equals labels
equals = np.equal(final_labels, y_test)
# Accuracy
acc = np.mean(equals.astype(int))

# Show accuracy
print('Accuracy: ', acc)

# Point to predict
pred = [0.5, 0.5, 0]

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot training data
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train)
# Plot point to predict
ax.scatter(pred[0], pred[1], pred[2], marker='^', linewidths=8)
# Title
ax.set_title('Point to predict')
# Axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# Distances bewtween k points and pred
distances = distances_func(X_train, pred, y_train, k)

# Count number of points in each class
unique, counts = np.unique(distances[:, -1], return_counts=True)

# Choose class with more points around
final_label = unique[np.argmax(counts)]

# Colors for points
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# Select the number of colors equal to the number of classes
colors = colors[:n_class]

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
for i in range(n_class):
    # Only point for current class
    data = X_train[y_train == i]
    # Plot training data for current class
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=colors[i], label=i)
# Plot colored prediction
ax.scatter(pred[0], pred[1], pred[2], c=colors[int(final_label)], marker='^', linewidths=8)
# Title
ax.set_title('Prediction')
# Axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Axes limit
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 2)
ax.set_zlim(-2, 3)
# Choose nearest k points
pred_points = X_train[distances[:, 1].astype(np.int32)]
preds_labels = y_train[distances[:, 1].astype(np.int32)]
for i in range(n_class):
    # Only the nearest k points for current class
    data = pred_points[preds_labels == i]
    # Plot the nearest k points for current class
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=colors[i], label=i)
# Pot predicion
ax.scatter(pred[0], pred[1], pred[2], c=colors[int(final_label)], marker='^', linewidths=5)
# Title
ax.set_title('Nearest k points')
# Axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()

# SKLEARN #

# Create KNN model
knn = KNeighborsClassifier(21)
# Train model
knn.fit(X_train, y_train)

# Predcition
predict = knn.predict(np.reshape(pred, (1, -1)))

# Predicted class
print('Sklearn prediction: ', predict[0])

plt.show()
