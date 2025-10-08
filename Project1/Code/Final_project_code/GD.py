# %%
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Cost function
def mse(theta, X, y, lmbda=0):
    y_pred = X @ theta
    # Calculating MSE
    mse = jnp.mean((y - y_pred) ** 2)
    # If lmbda = 0 we calculate for OLS, else Ridge.
    ridge_penalty = lmbda * jnp.sum(theta**2) / len(y)
    return mse + ridge_penalty


# %% [markdown]
# # Gradient Descent

# %%
def gradient_descent( X, y, lmbda=0, eta=0.01, num_iters=1000, plot=True, font_size = 14):
    # Intital guess 0
    theta = jnp.zeros(X.shape[1])

    # Calculating the gradient of our cost function
    gradient_fn = jax.grad(mse, argnums=0)

    # Empty list for plotting
    cost_history = []

    for t in range(num_iters):
        # Calculating the gradient with current value of theta
        gradient = gradient_fn(theta, X, y, lmbda)
        # Update theta
        theta = theta - eta * gradient
        # Calculating the cost
        cost = mse(theta, X, y, lmbda)
        # Adding the cost to our plotting list
        cost_history.append(cost)

    cost_history = jnp.array(cost_history)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(cost_history, label="Cost")
        plt.xlabel("Iteration", fontsize=font_size)
        plt.ylabel("Cost($\\theta$)", fontsize=font_size)
        plt.title(f"Gradient Descent Convergence - $\\eta$ = {eta}", fontsize=(font_size+2))
        plt.legend(fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(True)
        plt.show()

    return theta, cost_history



# Gradient Descent with Momentum
def gradient_descent_momentum(X, y, lmbda=0, alpha=0.9, eta=0.01, n_iterations=1000, tolerance=1e-6, plot=False,  font_size = 14, method=None):

    # Extract number of datapoints and number of parameters from the design matrix
    n, p = X.shape
    # Intital guess 0
    theta = np.zeros((p, 1))

    # Initial velocity
    v = np.zeros((p, 1))

    # Empty list for plotting
    cost_history = []

    for i in range(n_iterations):
        # Compute gradient based on selected method
        if method=="ridge":
            gradient = (2/n) * X.T @ (X @ theta - y) + 2*lmbda*theta
        elif method=="lasso":
            gradient = (2/n) * X.T @ (X @ theta - y) + lmbda*np.sign(theta)
        else:
            gradient = (2/n) * X.T @ (X @ theta - y)
        
        # Update velocity and theta
        v = alpha * v - eta * gradient
        theta_new = theta + v
        
        # Check convergence
        if np.linalg.norm(theta_new - theta) < tolerance:
            break
        
        # Update theta
        theta = theta_new

        # Calculating the cost
        cost = mse(theta, X, y, lmbda=0)
        # Adding the cost to our plotting list
        cost_history.append(cost)
    
    cost_history = jnp.array(cost_history)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(cost_history, label="Cost")
        plt.xlabel("Iteration", fontsize=font_size)
        plt.ylabel("Cost($\\theta$)", fontsize=font_size)
        plt.title(f"Gradient Descent Convergence - $\\eta$ = {eta}", fontsize=(font_size+2))
        plt.legend(fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(True)
        plt.show()

    # Return optimal theta
    return theta, cost_history


# %% [markdown]
# # Stochastic Gradient Descent

# %%
def stochastic_gradient_descent(X, y, lmbda=0, eta=0.01, epochs=50, batch_size=5, plot=True, font_size=14):
    n, m = X.shape
    # Intital guess 0
    theta = jnp.zeros(m)

    # Empty list for plotting for each epoch
    cost_history_epoch = []
    cost_history_iter = []
    # Gradient function of mse w.r.t theta
    gradient_fn = jax.grad(mse, argnums=0)

    # PRNG key for reproducibility
    key = jax.random.PRNGKey(0)

    for epoch in range(epochs):
        # Shuffle dataset
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n)
        X_shuffled, y_shuffled = X[perm], y[perm]

        # Iterating through all mini-batches
        for i in range(0, n, batch_size):
            #Creating mini-batches
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Compute gradient for mini-batch
            grad = gradient_fn(theta, X_batch, y_batch, lmbda)

            # Update theta
            theta = theta - eta * grad
            
        # Compute cost for full dataset after each epoch
        cost = mse(theta, X, y, lmbda)

        # Add to history
        cost_history_epoch.append(cost)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(cost_history_epoch, label=f"SGD (batch={batch_size}) Cost")
        plt.xlabel("Epoch", fontsize=font_size)
        plt.ylabel("Cost($\\theta$)", fontsize=font_size)
        plt.title(f"SGD Convergence - $\\eta$ = {eta}, batch={batch_size}", fontsize=(font_size))
        plt.legend(fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(True)
        plt.show()

    return theta, cost_history_epoch
