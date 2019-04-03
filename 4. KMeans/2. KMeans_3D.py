from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_blobs

# MANUAL #

# Seed
np.random.seed(101)

# HyperParameters
num_features = 3
num_clusters = 4
num_points = 150

# Generate data
x, y_true = make_blobs(n_samples=num_points, n_features=num_features, centers=num_clusters)

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot data
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y_true)
# Title
ax.set_title('Data')
# Axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# Add empty column to data
new_points = np.hstack((x, np.zeros((len(x), 1))))

# Color for clusters
clusters_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# Take only as many colors as there are classes
clusters_colors = clusters_colors[:num_clusters]


def distance(clusters, points):
    """Compute distance."""
    # Empty list for distances
    distances = np.empty((num_clusters, 1))
    for i in range(len(points)):
        for n in range(num_clusters):
            # Distance between each points and clusters
            distances[n] = np.linalg.norm(clusters[n]-points[i, :-1])
        # Nearest cluster
        near_cluster = np.argmin(distances)
        # New column to points data
        points[i, -1] = near_cluster
    return points


def color_points(points):
    """Color points."""
    points_colors = []
    for i in range(len(points)):
        points_colors.append(clusters_colors[int(points[i, -1])])
    return points_colors


def move_centroid(clusters_pos, points):
    """Move centroids."""
    for n in range(num_clusters):
        # Points belonging to current cluster
        selected_points = [point[:-1] for point in points if n == point[-1]]
        if selected_points:
            # New cluster position
            clusters_pos[n] = np.mean(selected_points, axis=0)
    return clusters_pos


def error_func(clusters, points):
    """Compute errors (similar as distance's function)."""
    errors = np.empty((num_clusters))
    for n in range(num_clusters):
        # Distance between cluster and its points
        distances = [np.linalg.norm(clusters[n]-point[:-1]) for point in points if n == point[-1]]
        errors[n] = np.sum(distances)
    return np.sum(errors)


# Minimal error
min_error = float(np.inf)

# Train algorithm 10 times and choose the best result
for i in range(10):
    # Initialize clusters randomly
    clusters = np.random.uniform(x.min(), x.max(), size=(num_clusters, num_features))
    # Training loop
    for epoch in range(100):
        # Distances
        points = distance(clusters, new_points)
        # Color points
        points_colors = color_points(points)
        # Move centroids
        clusters = move_centroid(clusters, new_points)
    # Error
    error = error_func(clusters, new_points)
    # If error < min_error save result
    if error < min_error:
        min_error = error
        save_clusters = clusters
    print(error)

# Distances
points = distance(save_clusters, new_points)
# Color points
points_colors = color_points(points)

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot data
ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], c=points_colors)
# Plot clusters
ax.scatter(save_clusters[:, 0], save_clusters[:, 1], save_clusters[:, 2],
           c=clusters_colors, marker='^', linewidths=10)
# Title
ax.set_title('Final result: \n' + 'Error: ' + str(min_error))
# Axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# SKLEARN #

# Create KMeans model
km = KMeans(n_clusters=4)
# Train model
km.fit(new_points[:, :-1])

# Distances
points = distance(km.cluster_centers_, new_points)
# Color points
points_colors = color_points(points)

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot data
ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], c=points_colors)
# Plot clusters
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], km.cluster_centers_[
           :, 2], c=clusters_colors, marker='^', linewidths=10)
# Title
ax.set_title('Final result - Sklearn')
# Axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.show()
