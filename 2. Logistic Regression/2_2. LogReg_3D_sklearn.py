import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

# Seed
np.random.seed(101)

# HyperParameters
num_classes = 2
num_features = 3
num_points = 500
epochs = 50000
learning_rate = 0.001

# Generate data
x, y_true = make_classification(n_samples=num_points, n_features=num_features, n_redundant=0, n_clusters_per_class=num_classes, class_sep=1.5)

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

# Create Logistic Regression model
log = LogisticRegression()
# Train model
log.fit(X_train, y_train)

# Predictions
preds = log.predict(X_test)

# Classification report
print(classification_report(y_test, preds))
# Confusion matrix
print(confusion_matrix(y_test, preds))

# Weights and bias from model
W1 = log.coef_[0][0]
W2 = log.coef_[0][1]
W3 = log.coef_[0][2]
b = log.intercept_[0]

# Function to create Decision Boundary (seperating surface)
z = lambda x1,x2: (-b - W1*x1 - W2*x2) / W3

# Data to create Decision Boundary
tmp = np.linspace(X_test.min()-1,X_test.max()+1,51)
xx1,xx2 = np.meshgrid(tmp,tmp)

# Create figure
fig = plt.figure()
ax = Axes3D(fig)
# Plot predictions
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=preds)
# Plot Decision Boundary
ax.plot_surface(xx1, xx2, z(xx1,xx2), color = None, alpha = 0.5)
# Title
ax.set_title('Decision Boundary')
# Axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

# Create figure
fig = plt.figure()

# FPR and TPR
fpr_sklearn, tpr_sklearn, thresholds = roc_curve(y_test, preds)
# AUC
roc_auc = auc(fpr_sklearn, tpr_sklearn)

# Plot ROC Curve
plt.plot(fpr_sklearn, tpr_sklearn, color='darkorange',
         lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.show()