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
def cross_validation(x, y, degrees=5, lmbda=0, lass=False, n_folds=5, LOO=False):
    
    results_list = []

    if LOO:
        folds = LeaveOneOut()
    else:
        folds = KFold(n_folds, shuffle=True, random_state=42)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    poly_features = PolynomialFeatures(degrees)
    
    for i, (train_indx, test_indx) in enumerate(folds.split(x)):
        x_train, x_test, y_train, y_test = x[train_indx], x[test_indx], y[train_indx], y[test_indx]

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        X_train = poly_features.fit_transform(x_train)
        X_test = poly_features.transform(x_test)

        if lass:
            result = lasso_reg_mod(X_train, X_test, y_train, y_test, lmbda=lmbda, eta=0.01, n_iter=500, GD="GD")
        else:
            result = fit_polynomial_mod(X_train, X_test, y_train, y_test, lmbda=lmbda, n_iter=500, GD=True, momentum=False)

        results_list.append(result)

    avg_results = {
        "train_mse": np.mean([r["train_mse"] for r in results_list]),
        "test_mse": np.mean([r["test_mse"] for r in results_list]),
        "train_r2": np.mean([r["train_r2"] for r in results_list]),
        "test_r2": np.mean([r["test_r2"] for r in results_list])
    }

    return avg_results


