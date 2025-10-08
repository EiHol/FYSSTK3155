from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from bootstrap import bootstrap


def bias_variance_tradeoff(x, y, degrees, bootstraps=500, test_size=0.3, random_state=27):
    #y_true is our noiseless function

    biases = []
    variances = []
    mses = []

    for p in range(degrees):
        # Create polynomial features with degree p
        X = PolynomialFeatures(degree=p).fit_transform(x.reshape(-1, 1))

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Empty array to store predictions for each bootstrap
        predictions = np.zeros((bootstraps, len(y_test)))

        X_train_re = bootstrap(X_train, n_samples=len(X_train), n_bootstraps=bootstraps)
        y_train_re = bootstrap(y_train, n_samples=len(X_train), n_bootstraps=bootstraps)

        for b, (X_tr_re, y_tr_re) in enumerate(zip(X_train_re, y_train_re)):

            # Fit model on the resampled training data
            model = LinearRegression()
            model.fit(X_tr_re, y_tr_re)

            # Make predictions on the fixed test data
            predictions[b, :] = model.predict(X_test)

        # Calculate the bias^2, variance and mse
        y_pred_mean = np.mean(predictions, axis=0)
        bias = np.mean((y_test - y_pred_mean)**2)
        variance = np.mean(np.var(predictions, axis=0))
        mse = np.mean((y_test.reshape(1, -1) - predictions)**2)

        # Store results for this polynomial degree
        biases.append(bias)
        variances.append(variance)
        mses.append(mse)

    
    return biases, variances, mses