# %%
from GD import gradient_descent, stochastic_gradient_descent, gradient_descent_momentum, mse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.utils import resample



# %%
# Generate data
# Generate data
def generate_data(n, noise = True, seed=42):
    # Fixed seed value to ensure consisten results across runs
    np.random.seed(seed)
    # Creating an array with equally spaced data
    x = np.linspace(-1, 1, n)
    # Runges equation with noise
    if noise:
        y = 1 / (1 + 25*x**2) +  np.random.normal(0, 0.1, size=n)
    else:
        y = 1 / (1 + 25*x**2)
    return x, y



# %%
# Design matrix
def polynomial_features(x, p, lmbda):
    # Length of the design matrix
    n = len(x)
    # Creating an empty feature matrix
    if lmbda == 0: #OLS
        X = np.zeros((n, p + 1))
        # Intercept column
        X[:, 0] = np.ones(n)
        for power in range(1, p+1):
            X[:, power] = x**power

    else: # Ridge
        X = np.zeros((n, p))
        for power in range(1, p+1): 
            X[:, power-1] = x**power
            
    
    return X


# %%
# Design matrix
def polynomial_features(x, p, lmbda):
    # Length of the design matrix
    n = len(x)
    # Creating an empty feature matrix
    if lmbda == 0: #OLS
        X = np.zeros((n, p + 1))
        # Intercept column
        X[:, 0] = np.ones(n)
        for power in range(1, p+1):
            X[:, power] = x**power

    else: # Ridge
        X = np.zeros((n, p))
        for power in range(1, p+1): 
            X[:, power-1] = x**power
            
    
    return X


def fit_polynomial(x, y, degree, lmbda = 0, test_size=0.3, eta = 0.01, epochs = 50, batch_size = 5, GD=False, n_iter = None, SGD = False, momentum = False):
    # Creating the feature matrix
    Xpoly = polynomial_features(x, degree, lmbda)
    X_train, X_test, y_train, y_test = train_test_split(Xpoly, y, test_size=test_size, random_state = 10)   
    
    if lmbda == 0: #OLS 
        # Getting the estimators
        if GD: #Vanilla GD
            theta, cost_hist = gradient_descent(X_train, y_train, lmbda , eta, num_iters=n_iter, plot = False)
        elif SGD: #Stochastic GD
            theta, cost_hist = stochastic_gradient_descent(X_train, y_train, lmbda, eta=eta, epochs=epochs, batch_size=batch_size, plot = False)
        elif momentum: #GD w/momentum
            theta, cost_hist = gradient_descent_momentum(X_train, y_train, eta=eta, n_iterations=n_iter, plot=False)
        else:
            theta= np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

        # Creating Prediction models
        train_pred = X_train @ theta
        test_pred = X_test @ theta

        # Storing the different vectors which we will use for our plots
        results = { "train_mse": mean_squared_error(y_train, train_pred),
                    "test_mse": mean_squared_error(y_test, test_pred),
                    "train_r2": r2_score(y_train, train_pred),
                    "test_r2": r2_score(y_test, test_pred),
                    "theta": theta,}

    else: #Ridge
        # Scale X
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Center y
        y_offset = np.mean(y_train)
        y_train_c = y_train - y_offset

        # Getting the estimators
        if GD: #Vanilla GD
            theta, cost_hist = gradient_descent(X_train_s, y_train_c, lmbda , eta, num_iters=n_iter, plot=False)
        elif SGD: #Stochastic GD
            theta, cost_hist = gradient_descent(X_train_s, y_train_c, lmbda, eta, epochs=epochs, batch_size=batch_size, plot = False)
        elif momentum: #GD w/momentum
            theta, cost_hist = gradient_descent_momentum(X_train, y_train, eta=eta, n_iterations=n_iter, plot=False)
        else:
            _, p = X_train_s.shape
            theta = np.linalg.inv(X_train_s.T @ X_train_s + lmbda * np.eye(p)) @ X_train_s.T @ y_train_c

        # Predictions (add offset back)
        y_train_pred = X_train_s @ theta + y_offset
        y_test_pred  = X_test_s @ theta + y_offset

        # Storing the different vectors which we will use for our plots
        results = {
                "train_mse": mean_squared_error(y_train, y_train_pred),
                "test_mse": mean_squared_error(y_test, y_test_pred),
                "train_r2": r2_score(y_train, y_train_pred),
                "test_r2": r2_score(y_test, y_test_pred),
                "theta": theta,}
    return results

# %%
def fit_polynomial_mod(X_train, X_test, y_train, y_test, lmbda = 0, test_size=0.3, eta = 0.01, epochs = 50, batch_size = 5, GD=False, n_iter = 1000, SGD = False, momentum = False):  
    if lmbda == 0: #OLS 
        # Getting the estimators
        if GD: #Vanilla GD
            theta, cost_hist = gradient_descent(X_train, y_train, lmbda , eta, num_iters=n_iter, plot = False)
        elif SGD: #Stochastic GD
            theta, cost_hist = stochastic_gradient_descent(X_train, y_train, lmbda, eta=eta, epochs=epochs, batch_size=batch_size, plot = False)
        elif momentum: #GD w/momentum
            theta, cost_hist = gradient_descent_momentum(X_train, y_train, eta=eta, n_iterations=n_iter, plot=False)
        else:
            theta= np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

        # Creating Prediction models
        train_pred = X_train @ theta
        test_pred = X_test @ theta

        # Storing the different vectors which we will use for our plots
        results = {"train_mse": mean_squared_error(y_train, train_pred),
                    "test_mse": mean_squared_error(y_test, test_pred),
                    "train_r2": r2_score(y_train, train_pred),
                    "test_r2": r2_score(y_test, test_pred),
                    "theta": theta,}

    else: #Ridge
        # Scale X
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Center y
        y_offset = np.mean(y_train)
        y_train_c = y_train - y_offset

        # Getting the estimators
        if GD: #Vanilla GD
            theta, cost_hist = gradient_descent(X_train_s, y_train_c, lmbda , eta, num_iters=n_iter, plot=False)
        elif SGD: #Stochastic GD
            theta, cost_hist = gradient_descent(X_train_s, y_train_c, lmbda, eta, epochs=epochs, batch_size=batch_size, plot = False)
        elif momentum: #GD w/momentum
            theta, cost_hist = gradient_descent_momentum(X_train_s, y_train_c, lmbda, eta=eta, n_iterations=n_iter, plot=False, method=None)
        else:
            _, p = X_train_s.shape
            theta = np.linalg.inv(X_train_s.T @ X_train_s + lmbda * np.eye(p)) @ X_train_s.T @ y_train_c

        # Predictions (add offset back)
        y_train_pred = X_train_s @ theta + y_offset
        y_test_pred  = X_test_s @ theta + y_offset

        # Storing the different vectors which we will use for our plots
        results = {
                "train_mse": mean_squared_error(y_train, y_train_pred),
                "test_mse": mean_squared_error(y_test, y_test_pred),
                "train_r2": r2_score(y_train, y_train_pred),
                "test_r2": r2_score(y_test, y_test_pred),
                "theta": theta,}
        
    return results


# %%
def plot_mse_r2(results, n, GD=False, R2=True):
    degrees   = [r["degree"] for r in results]
    train_mse = [r["train_mse"] for r in results]
    test_mse  = [r["test_mse"] for r in results]

    _, ax1 = plt.subplots(figsize=(8,5))
    ax1.set_xlabel("Polynomial Degree", fontsize=14)
    ax1.set_ylabel("MSE", fontsize=14)
    ax1.plot(degrees, train_mse, 'o-', color="tab:blue", label="Train MSE")
    ax1.plot(degrees, test_mse,  's-', color="tab:red",  label="Test MSE")

    if R2:  # Option of having MSE vs Polynomial degree
        train_r2 = [r["train_r2"] for r in results]
        test_r2  = [r["test_r2"] for r in results]
        ax2 = ax1.twinx()
        ax2.set_ylabel("R²", fontsize=14)
        ax2.plot(degrees, train_r2, 'o-', color="tab:green", label="Train R²")
        ax2.plot(degrees, test_r2,  's-', color="tab:orange", label="Test R²")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    else:
        ax1.legend(loc="upper right")
    
    title = rf"MSE {'& $R^2$ ' if R2 else ''}{'w/GD ' if GD else ''}(n={n})"
    plt.title(title, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()

    # %%
def plot_mse_r2_Lasso(results, n, GD="GD", R2=True):
    degrees   = [r["degree"] for r in results]
    train_mse = [r["train_mse"] for r in results]
    test_mse  = [r["test_mse"] for r in results]

    _, ax1 = plt.subplots(figsize=(8,5))
    ax1.set_xlabel("Polynomial Degree", fontsize=14)
    ax1.set_ylabel("MSE", fontsize=14)
    ax1.plot(degrees, train_mse, 'o-', color="tab:blue", label="Train MSE")
    ax1.plot(degrees, test_mse,  's-', color="tab:red",  label="Test MSE")

    if R2:  # Option of having MSE vs Polynomial degree
        train_r2 = [r["train_r2"] for r in results]
        test_r2  = [r["test_r2"] for r in results]
        ax2 = ax1.twinx()
        ax2.set_ylabel("R²", fontsize=14)
        ax2.plot(degrees, train_r2, 'o-', color="tab:green", label="Train R²")
        ax2.plot(degrees, test_r2,  's-', color="tab:orange", label="Test R²")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    else:
        ax1.legend(loc="upper right")
    
    title = f"MSE {'& $R^2$ ' if R2 else ''}w/{GD} (n={n})"
    plt.title(title, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()

# %%
def plot_mse_r2_CV(results, n, GD="GD", R2=True):
    kfolds   = [r["K-Folds"] for r in results]
    train_mse = [r["train_mse"] for r in results]
    test_mse  = [r["test_mse"] for r in results]

    _, ax1 = plt.subplots(figsize=(8,5))
    ax1.set_xlabel("K-Folds", fontsize=14)
    ax1.set_ylabel("Mean MSE", fontsize=14)
    ax1.plot(kfolds, train_mse, 'o-', color="tab:blue", label="Train MSE")
    ax1.plot(kfolds, test_mse,  's-', color="tab:red",  label="Test MSE")

    if R2:  # Option of having MSE vs Polynomial degree
        train_r2 = [r["train_r2"] for r in results]
        test_r2  = [r["test_r2"] for r in results]
        ax2 = ax1.twinx()
        ax2.set_ylabel("R²", fontsize=14)
        ax2.plot(kfolds, train_r2, 'o-', color="tab:green", label="Train R²")
        ax2.plot(kfolds, test_r2,  's-', color="tab:orange", label="Test R²")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    else:
        ax1.legend(loc="upper right")
    
    title = f"Mean MSE {'& $R^2$ ' if R2 else ''}w/{GD} (n={n})"
    plt.title(title, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()



# %%
def mse_heatmap_eta_degree(results_by_eta, degrees,n,n_iter):
    # Extract eta values
    eta_values = list(results_by_eta.keys())
    # Initialize matrix: rows = degrees, cols = eta
    mse_matrix = np.zeros((len(degrees), len(eta_values)))

    # Fill matrix
    for i, d in enumerate(degrees):
        for j, eta in enumerate(eta_values):
            res = next(r for r in results_by_eta[eta] if r["degree"] == d)
            mse_matrix[i, j] = res["test_mse"]

    # Plot heatmap
    plt.figure(figsize=(8,5))
    im = plt.imshow(mse_matrix, cmap='jet', origin='lower', aspect='auto')

    plt.xticks(ticks=np.arange(len(eta_values)), labels=[f"{e:.5f}" for e in eta_values], fontsize=12)
    plt.yticks(ticks=np.arange(len(degrees)), labels=degrees, fontsize=12)
    plt.xlabel("Learning Rate (η)", fontsize=14)
    plt.ylabel("Polynomial Degree", fontsize=14)
    plt.title(f"MSE Heatmap (n ={n}, n_iter = {n_iter})", fontsize=15)
    plt.colorbar(im, label='Test MSE')

    # Overlay text values
    for i in range(len(degrees)):
        for j in range(len(eta_values)):
            plt.text(j, i, f"{mse_matrix[i,j]:.5f}", ha="center", va="center", color="w", fontsize=9)

    plt.tight_layout()
    plt.show()

    return mse_matrix

def plot_theta_heatmap(thetas, degrees):

    # Pad theta vectors to equal length
    max_len = max(len(t) for t in thetas)
    theta_matrix = np.full((len(degrees), max_len), np.nan)
    
    for i, t in enumerate(thetas):
        theta_matrix[i, :len(t)] = t

    plt.figure(figsize=(8, 5))
    plt.imshow(theta_matrix, aspect="auto", cmap="coolwarm", interpolation="nearest")
    
    plt.colorbar(label="Coefficient value")
    plt.xlabel("Coefficient index (θ₀, θ₁, …)", fontsize=13)
    plt.ylabel("Polynomial degree", fontsize=13)
    plt.title("OLS coefficients vs polynomial degree", fontsize=14)
    
    plt.xticks(range(max_len), [f"$\\theta_{{{i}}}$" for i in range(max_len)])
    plt.yticks(range(len(degrees)), degrees)
    
    plt.tight_layout()
    plt.show()

def plot_metrics(results, n, GD=False, metric="both", lam=0, eta=None):
    degrees = [r["degree"] for r in results]
    _, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Polynomial Degree", fontsize=14)

    # --- Plot MSE ---
    if metric in ("mse", "both"):
        train_mse = [r["train_mse"] for r in results]
        test_mse  = [r["test_mse"]  for r in results]
        ax1.set_ylabel("MSE", fontsize=14)
        ax1.plot(degrees, train_mse, '--o', color="tab:blue", label="Train MSE")
        ax1.plot(degrees, test_mse,  '-s', color="tab:red",  label="Test MSE")

    # --- Plot R² ---
    if metric in ("r2", "both"):
        train_r2 = [r["train_r2"] for r in results]
        test_r2  = [r["test_r2"]  for r in results]
        if metric == "both":
            ax2 = ax1.twinx()
            ax2.set_ylabel("R²", fontsize=14)
            ax2.plot(degrees, train_r2, '--o', color="tab:green", label="Train R²")
            ax2.plot(degrees, test_r2,  '-s', color="tab:orange", label="Test R²")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_led_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
        else:
            ax1.set_ylabel("R²", fontsize=14)
            ax1.plot(degrees, train_r2, '--o', color="tab:green", label="Train R²")
            ax1.plot(degrees, test_r2,  '-s', color="tab:orange", label="Test R²")
            ax1.legend(loc="upper right")
    else:
        ax1.legend(loc="upper right")

    # --- Title formatting ---
    metric_title = {"mse": "MSE", "r2": "$R^2$", "both": "MSE & $R^2$"}[metric]
    lam_text = f", λ={lam}" if lam != 0 else ""
    eta_text = f", η={eta}" if eta is not None else ""
    title = f"{metric_title}{' w/ GD' if GD else ''} (n={n}{lam_text}{eta_text})"

    plt.title(title, fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()



# %%
def mse_heatmap(results_by_n, degrees, GD=False):
    # Build matrix: 
    # rows = degrees, cols = n
    n_values = list(results_by_n.keys())
    # Empty matrix
    mse_matrix = np.zeros((len(degrees), len(n_values)))
    
    for i, d in enumerate(degrees):
        for j, n in enumerate(n_values):
            res = next(r for r in results_by_n[n] if r["degree"] == d)
            mse_matrix[i, j] = res["test_mse"]

    # Plotting the heatmap
    plt.figure(figsize=(8,5))
    im = plt.imshow(mse_matrix, cmap='jet', origin='lower', aspect='auto')

    plt.xticks(ticks=np.arange(len(n_values)), labels=n_values, fontsize=14)
    plt.yticks(ticks=np.arange(len(degrees)),labels=degrees, fontsize=14)
    plt.xlabel("Number of Data Points (n)", fontsize=14)
    plt.ylabel("Polynomial Degree", fontsize=14)
    if GD:
        plt.title("OLS w/GD MSE Heatmap",fontsize=14)
    else:
        plt.title("OLS MSE Heatmap",fontsize=14)
    plt.colorbar(im, label='Test MSE')

    # Placing MSE values in the cells
    for i in range(len(degrees)):
        for j in range(len(n_values)):
            plt.text(j, i, f"{mse_matrix[i,j]:.3f}", ha="center", va="center", color="w", fontsize=10)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()
    
    return mse_matrix

def plot_theta_heatmap_ols(thetas, degrees, n):
    import numpy as np
    import matplotlib.pyplot as plt

    # Pad theta vectors to equal length (excluding intercept)
    max_len = max(len(t) - 1 for t in thetas)
    theta_matrix = np.full((len(degrees), max_len), np.nan)

    for i, t in enumerate(thetas):
        theta_matrix[i, :len(t)-1] = t[1:]   # Skip intercept (theta_0)

    # Plot heatmap
    plt.figure(figsize=(8, 5))
    plt.imshow(theta_matrix, aspect="auto", cmap="coolwarm", interpolation="nearest")

    # Colorbar and labels
    plt.colorbar(label="Coefficient value")
    plt.xlabel(r"Coefficient index ($\theta_i$)", fontsize=13)
    plt.ylabel("Polynomial degree", fontsize=13)
    plt.title(f"OLS coefficients vs polynomial degree (n = {n})", fontsize=14)

    # Adjust ticks (shift indices by 1 since intercept is removed)
    plt.xticks(range(max_len), [f"$\\theta_{{{i}}}$" for i in range(1, max_len + 1)])
    plt.yticks(range(len(degrees)), degrees)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()


def plot_theta_heatmap_ridge(thetas, degrees, n, lmbda):
    max_len = max(len(t) for t in thetas)
    theta_matrix = np.full((len(degrees), max_len), np.nan)
    
    for i, t in enumerate(thetas):
        theta_matrix[i, :len(t)] = t

    plt.figure(figsize=(8, 5))
    plt.imshow(theta_matrix, aspect="auto", cmap="coolwarm", interpolation="nearest")
    
    plt.colorbar(label="Coefficient value")
    plt.xlabel(r"Coefficient index ($\theta_i$)", fontsize=13)
    plt.ylabel("Polynomial degree", fontsize=13)
    plt.title(rf"Ridge coefficients vs polynomial degree (n = {n}, $\lambda = {lmbda}$)", fontsize=14)
    
    plt.xticks(range(max_len), [f"$\\theta_{{{i}}}$" for i in range(max_len)])
    plt.yticks(range(len(degrees)), degrees)
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()





# %%
#Running experiments for our plotting
def run_experiments(x, y, degrees, lmbda = 0, test_size=0.3, eta = 0.01, epochs = 50, batch_size = 5, GD=False, n_iter = 1000, SGD = False, momentum = False):
    # Empty list of results
    results = []
    for d in degrees:
        res = fit_polynomial(x, y, d, lmbda, test_size, eta, epochs, batch_size, GD, n_iter, SGD, momentum )
        res["degree"] = d
        results.append(res)
    return results


