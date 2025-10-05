# %% [markdown]
# Cross-Validation

# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures
from OLS_Ridge import fit_polynomial_mod

# %%
# Cross validation
def cross_validation(x, y, degrees=5, n_folds=5, LOO=False, method=None, lmbda = 0, test_size=0.3, eta = 0.01, epochs = 50, batch_size = 5, GD=False, n_iter = 1000, SGD = False, momentum = False):
    
    if LOO:
        # Leave one out CV
        folds = LeaveOneOut()
    else:
        # Initiate n folds
        folds = KFold(n_folds)

    # Ensure x is 2D
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Create polynomial features
    poly_features = PolynomialFeatures(degrees)
    
    # Split the date into n_folds
    for i, (train_indx, test_indx) in enumerate(folds.split(x)):
        # Extract training and testing data
        x_train, x_test, y_train, y_test = x[train_indx], x[test_indx], y[train_indx], y[test_indx]

        # Reshape data
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # Fit the training and test data
        X_train = poly_features.fit_transform(x_train)
        X_test = poly_features.transform(x_test)

        results = fit_polynomial_mod(X_train, X_test, y_train, y_test, n_iter=1000, lmbda=lmbda, eta=eta, epochs=epochs, batch_size=batch_size, GD=GD, n_iter=n_iter, SGD=SGD, momentum=momentum)

    return results



