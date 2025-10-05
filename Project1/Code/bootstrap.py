# Imports
import numpy as np
from sklearn.utils import resample

def bootstrap(data, n_samples, n_bootstraps):
    # Resampling with replacement
    replacement = True

    # Store the datasets in an array
    # If data is 1D
    if data.ndim == 1:
        datasets = np.empty((n_bootstraps, n_samples))
    # If data is 2D
    else:
        datasets = np.empty((n_bootstraps, n_samples, data.shape[1]))

    for i in range(n_bootstraps):
        # Perform bootstrapping
        datasets[i] = resample(data, replace=replacement, n_samples=n_samples)

    # Return an array of bootstrap datasets
    return datasets