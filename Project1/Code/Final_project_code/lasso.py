# Imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from GD import gradient_descent_momentum

def lasso_reg(x, y, degree, eta=0.01, GD=None, lmbda=0.01, n_iter=1000, test_size=0.3):

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Setting up the design matrix
    poly_features = PolynomialFeatures(degree)
    X = poly_features.fit_transform(x)

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=10)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Center y
    y_offset = np.mean(y_train)
    y_train_c = y_train - y_offset

    # Compute theta based on selected GD method
    if GD == "GDM":
        theta, cost_history = gradient_descent_momentum(X_train_s, y_train_c, lmbda=lmbda, alpha=0.9, eta=eta, n_iterations=n_iter, tolerance=1e-6, plot=False,  font_size = 14, method="lasso")
    else:
        theta, cost_history = gradient_descent_momentum(X_train_s, y_train_c, lmbda=lmbda, alpha=0, eta=eta, n_iterations=n_iter, tolerance=1e-6, plot=False,  font_size = 14, method="lasso")

    # Making predictions and addign back the offset
    y_train_pred = X_train_s @ theta + y_offset
    y_test_pred = X_test_s @ theta + y_offset

    # Results
    results = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "theta": theta
    }

    return results


def lasso_reg_mod(X_train, X_test, y_train, y_test, eta=0.01, GD=None, lmbda=0.01, n_iter=1000, test_size=0.3):

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Center y
    y_offset = np.mean(y_train)
    y_train_c = y_train - y_offset

    # Compute theta based on selected GD method
    if GD == "GDM":
        theta, cost_history = gradient_descent_momentum(X_train_s, y_train_c, lmbda=lmbda, alpha=0.9, eta=eta, n_iterations=n_iter, tolerance=1e-6, plot=False,  font_size = 14, method="lasso")
    else:
        theta, cost_history = gradient_descent_momentum(X_train_s, y_train_c, lmbda=lmbda, alpha=0, eta=eta, n_iterations=n_iter, tolerance=1e-6, plot=False,  font_size = 14, method="lasso")

    # Making predictions
    y_train_pred = X_train_s @ theta + y_offset
    y_test_pred = X_test_s @ theta + y_offset

    # Results
    results = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "theta": theta
    }

    return results