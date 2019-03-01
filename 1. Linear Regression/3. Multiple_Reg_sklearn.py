import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# Merge x1 and x2
x_both = np.empty((len(x1),2))
x_both[:,0] = x1
x_both[:,1] = x2

# Create Linear Regression model
lr = LinearRegression()
# Train model
lr.fit(x_both, y_true)

# All combinations of x1 and x2 for plane
x1_surf, x2_surf = np.meshgrid(np.linspace(x1.min(), x1.max(), 100), np.linspace(x2.min(), x2.max(), 100))

# Merge x1_surf and x2_surf
x_both_surf = np.empty((len(x1_surf.ravel()),2))
x_both_surf[:,0] = x1_surf.ravel()
x_both_surf[:,1] = x2_surf.ravel()

# Predictions for x_both_surf
preds = lr.predict(x_both_surf)

# Create figure
fig = plt.figure()
ax = Axes3D(fig)

# Plot data
ax.scatter(x1, x2, y_true)
# Plot plane fitted to data
ax.plot_surface(x1_surf,x2_surf, preds.reshape(x1_surf.shape), color='None', alpha=0.5)
# Title
ax.set_title('Fitted Plane')
# Labels names
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y_true')

plt.show()