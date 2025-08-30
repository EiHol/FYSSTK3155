import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Functions from lecture notes
def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model)**2) / n

def OLS_parameters(X, y):
    """Compute OLS parameters using normal equation"""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def create_design_matrix(x, degree):
    """Create design matrix for polynomial of given degree"""
    n = len(x)
    X = np.zeros((n, degree + 1))
    for i in range(degree + 1):
        X[:, i] = x**i
    return X

# Set random seed for reproducibility (matching project materials)
np.random.seed(2018)

# Generate the same data as in the project materials
n_samples = 50
x = np.linspace(-3, 3, n_samples).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
# Flatten for our custom implementation
x_flat = x.flatten()

# Split data into training and test sets (using same test size as project materials)
x_train, x_test, y_train, y_test = train_test_split(x_flat, y.flatten(), test_size=0.2, random_state=2018)

# Range of polynomial degrees to test
degrees = range(1, 11)  # 1 to 10 as requested (2 to 10 would be range(2, 11))
train_mse = []
test_mse = []

print("Polynomial Degree Analysis:")
print("=" * 40)

# Loop through each polynomial degree
for degree in degrees:
    # Create design matrices for current polynomial degree
    X_train = create_design_matrix(x_train, degree)
    X_test = create_design_matrix(x_test, degree)
    
    # Train the model (find optimal parameters)
    beta = OLS_parameters(X_train, y_train)
    
    # Make predictions
    y_train_pred = X_train @ beta
    y_test_pred = X_test @ beta
    
    # Calculate MSE for both training and test data
    train_mse_current = MSE(y_train, y_train_pred)
    test_mse_current = MSE(y_test, y_test_pred)
    
    # Store results
    train_mse.append(train_mse_current)
    test_mse.append(test_mse_current)
    
    # Print results
    print(f"Degree {degree:2d}: Train MSE = {train_mse_current:.6f}, Test MSE = {test_mse_current:.6f}")

# Create the plot (reproducing Figure 2.11 style)
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_mse, 'o-', color='blue', label='Training Error', linewidth=2, markersize=6)
plt.plot(degrees, test_mse, 'o-', color='red', label='Test Error', linewidth=2, markersize=6)

plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Bias-Variance Tradeoff: Training vs Test Error', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale often used for MSE plots
plt.xticks(degrees)

# Add annotations to highlight key concepts
min_test_idx = np.argmin(test_mse)
optimal_degree = degrees[min_test_idx]
plt.axvline(x=optimal_degree, color='green', linestyle='--', alpha=0.7, 
           label=f'Optimal Complexity (Degree {optimal_degree})')
plt.legend(fontsize=12)

# Annotate regions
plt.annotate('Underfitting\n(High Bias)', xy=(2, max(test_mse[:3])), 
            xytext=(3, max(test_mse[:3])*2), 
            arrowprops=dict(arrowstyle='->', color='orange'),
            fontsize=10, ha='center')

plt.annotate('Overfitting\n(High Variance)', xy=(9, test_mse[-2]), 
            xytext=(7.5, test_mse[-2]*3), 
            arrowprops=dict(arrowstyle='->', color='orange'),
            fontsize=10, ha='center')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("SUMMARY:")
print("="*50)
print(f"Optimal polynomial degree: {optimal_degree}")
print(f"Minimum test MSE: {min(test_mse):.6f}")
print(f"Training MSE at optimal degree: {train_mse[min_test_idx]:.6f}")

# Additional analysis: show the bias-variance tradeoff concept
print(f"\nBias-Variance Tradeoff Analysis:")
print(f"- Low complexity (degree 1-2): High bias, low variance (underfitting)")
print(f"- Optimal complexity (degree {optimal_degree}): Good bias-variance balance")
print(f"- High complexity (degree 8-10): Low bias, high variance (overfitting)")

# Compare first and last degrees
print(f"\nComparison:")
print(f"Degree 1 - Train: {train_mse[0]:.6f}, Test: {test_mse[0]:.6f}")
print(f"Degree 10 - Train: {train_mse[-1]:.6f}, Test: {test_mse[-1]:.6f}")
print(f"Gap for degree 1: {abs(test_mse[0] - train_mse[0]):.6f}")
print(f"Gap for degree 10: {abs(test_mse[-1] - train_mse[-1]):.6f}")