import matplotlib.pyplot as plt
import numpy as np
from ransac import *

# Generate synthetic data for testing the RANSAC algorithm
n_size = 5000
m = 3  # True slope
data_x = np.linspace(2, 10, n_size)
data_y = m * data_x + np.random.normal(0, 0.1, n_size)  # Add noise to the data


# Generate outliers
n_size2 = 50
outliers_x = np.linspace(7, 10, n_size2)
outliers_y = 10 + np.random.normal(0, 0.01, n_size2)  # Outliers have a different slope

# Combine the data and outliers
comb_data_y = np.concatenate((data_y, outliers_y))
comb_data_x = np.concatenate((data_x, outliers_x))
input_data = np.stack((comb_data_x, comb_data_y), axis=1)

# Run RANSAC algorithm to find the best-fit model
bestFit, bestData = ransac(input_data=input_data, iterations=20, temp_points=10, thres=0.4, min_good_points=int(0.8 * n_size))

# Plotting the results
fig, ax = plt.subplots(ncols=2)

# Plot original data
ax[0].scatter(comb_data_x, comb_data_y, label="Data points")
ax[0].set(title="Original Data")

# Plot filtered data after RANSAC
if bestData is not None:
    ax[1].scatter(bestData[:, 0], bestData[:, 1], c="red", label="Inliers")
    ax[1].set(title="Filtered Data")

plt.show()
