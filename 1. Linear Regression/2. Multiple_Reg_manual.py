import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

# Seed
np.random.seed(101)

# Noise for x and y data
x1_noise = np.random.randn(300)
x2_noise = np.random.randn(300)
y_noise = np.random.randn(300)

# Create x data
x1 = np.linspace(0.0, 10.0, 300) + x1_noise
x2 = np.linspace(0.0, 10.0, 300) + x2_noise

# Create y data
y_true = 0.7 * x1 + 0.5 * x2 + 5 + y_noise

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot data
ax.scatter(x1, x2, y_true)
# Title
ax.set_title('Data')
# Labels names
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y_true')

# Function to compute MSE error
def mse_error(labels, predictions):
	"""
		labels: y_true
		predictions: y_hat
	"""
	return (np.sum(labels - predictions)**2)/len(labels)

# Function to compute gradient
def gradient(features1, features2, labels, predictions):
	"""
		features1: x1
		features2: x2
		labels: y_true
		predictions: y_hat
	"""
	w1_gradient = -np.sum(np.dot(features1, (labels - predictions)))*(2/len(labels))
	w2_gradient = -np.sum(np.dot(features2, (labels - predictions)))*(2/len(labels))
	b_gradient = -np.sum(labels - predictions)*(2/len(labels))
	return w1_gradient, w2_gradient, b_gradient

# Randomly initialize W1, W2 and b
W1 = np.random.randn()
W2 = np.random.randn()
b = np.random.randn()

# Learning rate
learning_rate = 0.0001
# Epochs
epochs = 100000

# Training loop
for epoch in range(epochs):
	# Predictions
	y_hat = W1*x1 + W2*x2 + b
	# Error
	error = mse_error(y_true, y_hat)
	# Gradient
	new_w1, new_w2, new_b = gradient(x1, x2, y_true, y_hat)
	# Update W1
	W1 = W1 - learning_rate * new_w1
	# Update W2
	W2 = W2 - learning_rate * new_w2
	# Update b
	b = b - learning_rate * new_b

	# Print training progress every 1000 epochs
	if epoch % 1000 == 0:
		print('Epoch: {}, Error: {}'.format(epoch, error))

# All combinations of x1 and x2 for plane
x1_surf, x2_surf = np.meshgrid(np.linspace(x1.min(), x1.max(), 100), np.linspace(x2.min(), x2.max(), 100))

# Predictions for plane
y_pred = W1*x1_surf.ravel() + W2*x2_surf.ravel() + b
# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot data
ax.scatter(x1,x2,y_true)
# Plot plane fitted to data
ax.plot_surface(x1_surf,x2_surf, y_pred.reshape(x1_surf.shape), color='None', alpha=0.5)
# Title
ax.set_title('Fitted Plane')
# Labels names
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y_true')

plt.show()