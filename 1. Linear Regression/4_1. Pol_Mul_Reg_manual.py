import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Seed
np.random.seed(101)

# Noise for x and y data
x1_noise = np.random.randn(300)
x2_noise = np.random.randn(300)
y_noise = np.random.randn(300)

# Create x data
x1 = np.linspace(0, 10, 300) + x1_noise
x2 = np.linspace(0, 10, 300) + x2_noise

# Create y data
y_true = -0.7 * x1 + 0.4*x2 - 0.5 * (x1**2) + 0.2 * (x2**2) - 2 * (x1*x2) + 5 + y_noise

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


def mse_error(labels, predictions):
    """
    Compute MSE error.

    labels: y_true
    predictions: y_hat

    """
    error = 0
    for label, pred in zip(labels, predictions):
        error += (label - pred) ** 2
    return error / float(len(labels))


def gradient(features1, features2, labels, predictions):
    """
    Compute gradient.

    features1: x1
    features2: x2
    labels: y_true
    predictions: y_hat

    """
    w1_gradient = 0
    w2_gradient = 0
    w3_gradient = 0
    w4_gradient = 0
    w5_gradient = 0
    b_gradient = 0
    N = float(len(features1))
    for feat1, feat2, label, pred in zip(features1, features2, labels, predictions):
        w1_gradient += -(2/N) * feat1 * (label - pred)
        w2_gradient += -(2/N) * feat2 * (label - pred)
        w3_gradient += -(2/N) * (feat1 ** 2) * (label - pred)
        w4_gradient += -(2/N) * (feat2 ** 2) * (label - pred)
        w5_gradient += -(2/N) * (feat1 * feat2) * (label - pred)
        b_gradient += -(2/N) * (label - pred)
    return w1_gradient, w2_gradient, w3_gradient, w4_gradient, w5_gradient, b_gradient


# Randomly initialize W1, W2, W3, W4, W5 and b
W1 = np.random.randn()
W2 = np.random.randn()
W3 = np.random.randn()
W4 = np.random.randn()
W5 = np.random.randn()
b = np.random.randn()

# Learning rate
learning_rate = 0.0001
# Epochs
epochs = 100000

# Training loop
for epoch in range(epochs):
    # Predictions
    y_hat = W1*x1 + W2*x2 + W3*(x1**2) + W4*(x2**2) + W5*(x1*x2) + b
    # Error
    error = mse_error(y_true, y_hat)
    # Gradient
    new_w1, new_w2, new_w3, new_w4, new_w5, new_b = gradient(x1, x2, y_true, y_hat)
    # Update W1
    W1 = W1 - learning_rate * new_w1
    # Update W2
    W2 = W2 - learning_rate * new_w2
    # Update W3
    W3 = W3 - learning_rate * new_w3
    # Update W4
    W4 = W4 - learning_rate * new_w4
    # Update W5
    W5 = W5 - learning_rate * new_w5
    # Update b
    b = b - learning_rate * new_b
    # Print training progress every 1000 epochs
    if epoch % 1000 == 0:
        print('Epoch: {}, Error: {}'.format(epoch, error))

# All combinations of x1 and x2 for plane
x1_surf, x2_surf = np.meshgrid(np.linspace(x1.min(), x1.max(), 100),
                               np.linspace(x2.min(), x2.max(), 100))

# Predictions for plane
y_pred = W1*x1_surf.ravel() + W2*x2_surf.ravel() + W3*(x1_surf.ravel()**2) + W4 * \
    x2_surf.ravel()**2 + W5*x1_surf.ravel()*x2_surf.ravel() + b

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot data
ax.scatter(x1, x2, y_true)
# Plot plane fitted to data
ax.plot_surface(x1_surf, x2_surf, y_pred.reshape(x1_surf.shape), color='None', alpha=0.5)
# Title
ax.set_title('Fitted Plane')
# Labels names
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y_true')

plt.show()
