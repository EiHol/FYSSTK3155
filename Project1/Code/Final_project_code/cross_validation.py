# %% [markdown]
# Cross-Validation

# %%
# Imports
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures
from OLS_Ridge import fit_polynomial_mod
from lasso import lasso_reg_mod

# %%
# Cross validation
def cross_validation(x, y, degrees=5, lmbda=0, lass=False, n_folds=5, LOO=False, GD_meth="GD"):
    # List for returnign results
    results_list = []

    # Checking which fold method is selected
    if LOO:
        folds = LeaveOneOut()
    else:
        folds = KFold(n_folds, shuffle=True, random_state=42)

    # Setting parameters based on GD input
    if GD_meth == "GDM":
        GD_poly = False
        momentum = True
        GD_m="GDM"
    else:
        GD_poly = True
        momentum = False
        GD_m="GD"


    # Checking dimentions to avoid missmatch
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Create polynomial features
    poly_features = PolynomialFeatures(degrees)
    
    # Extract indexes to split the input data into training and testing sets
    for i, (train_indx, test_indx) in enumerate(folds.split(x)):
        # Split data
        x_train, x_test, y_train, y_test = x[train_indx], x[test_indx], y[train_indx], y[test_indx]

        # Reshape to avoid missmatch
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # Fit data to the polynomial features
        X_train = poly_features.fit_transform(x_train)
        X_test = poly_features.transform(x_test)

        # Run predictions
        if lass:
            result = lasso_reg_mod(X_train, X_test, y_train, y_test, lmbda=lmbda, eta=0.01, n_iter=500, GD=GD_m)
        else:
            result = fit_polynomial_mod(X_train, X_test, y_train, y_test, lmbda=lmbda, n_iter=500, GD=GD_poly, momentum=momentum)

        # Append results to a list
        results_list.append(result)

    # Calculate the average result for each fold
    avg_results = {
        "train_mse": np.mean([r["train_mse"] for r in results_list]),
        "test_mse": np.mean([r["test_mse"] for r in results_list]),
        "train_r2": np.mean([r["train_r2"] for r in results_list]),
        "test_r2": np.mean([r["test_r2"] for r in results_list])
    }

    return avg_results


