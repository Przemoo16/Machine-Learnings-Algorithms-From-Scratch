import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Seed
np.random.seed(101)

# HyperParameters
num_classes = 2
num_features = 3
num_points = 500
epochs = 50000
learning_rate = 0.001

# Generate data
x, y_true = make_classification(n_samples=num_points, n_features=num_features,
                                n_redundant=0, n_clusters_per_class=num_classes, class_sep=1.5)

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y_true, test_size=0.3, random_state=101)

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot test data
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test)
# Title
ax.set_title('Test data')
# Axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')


def sigmoid(x):
    """Sigmoid function."""
    return 1/(1+np.exp(-x))


# Weights and bias
W1 = np.random.randn()
W2 = np.random.randn()
W3 = np.random.randn()
b = np.random.randn()


def logistic_loss(y_true, y_hat):
    """Loss function."""
    loss = -np.mean(y_true * np.log(y_hat) + (1-y_true) * np.log(1-y_hat))
    return loss


def gradient(y_true, y_hat, features):
    """Gradient function."""
    err = y_true - y_hat
    w1_gradient = np.sum(-np.dot(features[:, 0], err))*(1/len(y_true))
    w2_gradient = np.sum(-np.dot(features[:, 1], err))*(1/len(y_true))
    w3_gradient = np.sum(-np.dot(features[:, 2], err))*(1/len(y_true))
    b_gradient = np.sum(err)*(1/len(y_true))
    return w1_gradient, w2_gradient, w3_gradient, b_gradient


# Training loop
for epoch in range(epochs):
    # Predictions for training data
    y_hat_train = sigmoid(W1*X_train[:, 0] + W2*X_train[:, 1] + W3*X_train[:, 2] + b)
    # Error
    error = logistic_loss(y_train, y_hat_train)
    # Print training progress
    print('Epoch: {}, Error: {}'.format(epoch, error))
    # New weights and bias
    new_w1, new_w2, new_w3, new_b = gradient(y_train, y_hat_train, X_train)
    # Update W1
    W1 = W1 - learning_rate * new_w1
    # Update W2
    W2 = W2 - learning_rate * new_w2
    # Update W3
    W3 = W3 - learning_rate * new_w3
    # Update bias
    b = b - learning_rate * new_b

# Predictions for test data
y_hat_test = sigmoid(W1*X_test[:, 0] + W2*X_test[:, 1] + W3*X_test[:, 2] + b)

# Round predictions to 1 or 0
test_rounded = [1 if el >= 0.5 else 0 for el in y_hat_test]


def z(x1, x2):
    """Create Decision Boundary (seperating surface)."""
    return (-b - W1*x1 - W2*x2) / W3


# Data to create Decision Boundary
tmp = np.linspace(X_test.min()-1, X_test.max()+1, 51)
xx1, xx2 = np.meshgrid(tmp, tmp)

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot predictions
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=test_rounded)
# Plot Decision Boundary
ax.plot_surface(xx1, xx2, z(xx1, xx2), color=None, alpha=0.5)
# Title
ax.set_title('Decision Boundary')
# Axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

# Plot ROC Curve
thresholds = np.linspace(1, 0, 101)

ROC = np.zeros((101, 2))

for i in range(101):
    t = thresholds[i]

    # Classifier / label agree and disagreements for current threshold.
    TP_t = np.logical_and(y_hat_test > t, y_test == 1).sum()
    TN_t = np.logical_and(y_hat_test <= t, y_test == 0).sum()
    FP_t = np.logical_and(y_hat_test > t, y_test == 0).sum()
    FN_t = np.logical_and(y_hat_test <= t, y_test == 1).sum()

    # Compute false positive rate for current threshold.
    FPR_t = FP_t / float(FP_t + TN_t)
    ROC[i, 0] = FPR_t

    # Compute true positive rate for current threshold.
    TPR_t = TP_t / float(TP_t + FN_t)
    ROC[i, 1] = TPR_t

# AUC
AUC = 0.
for i in range(100):
    AUC += (ROC[i+1, 0]-ROC[i, 0]) * (ROC[i+1, 1]+ROC[i, 1])
AUC *= 0.5


# Plot the ROC curve.
fig = plt.figure(figsize=(6, 6))
plt.plot(ROC[:, 0], ROC[:, 1], lw=2)
plt.title('ROC curve, AUC = %.4f' % AUC)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.xlabel('$FPR(t)$')
plt.ylabel('$TPR(t)$')
plt.grid()

plt.show()
