# GSBR – Gradient Supported Basic Regressor

## Overview

GSBR (Gradient Supported Basic Regressor) is a lightweight, custom linear regression model implemented in Python. It supports optimization via gradient descent and includes regularization options such as L1 (Lasso), L2 (Ridge), and ElasticNet to prevent overfitting. The model can minimize various loss functions, including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). It handles both dense and sparse input matrices, incorporates early stopping for efficiency, and provides options for data shuffling and verbose logging during training.

This model is ideal for educational purposes, quick prototyping, or scenarios where a simple, interpretable regressor is needed without relying on heavy external libraries beyond NumPy and SciPy.

## Installation & Requirements

To use GSBR, ensure you have the required dependencies:

```bash
pip install numpy scipy
```

The model is part of the `BasicModels` package (assumed to be structured under a parent module). Import it as follows:

```python
from BasicModels.GSBR import BasicRegressor
```

## Mathematical Formulation

### Prediction Function
The model predicts target values using a linear combination of features:

\[
\hat{y} = Xw + b
\]

Where:
- \(X\) is the input feature matrix,
- \(w\) is the weight vector,
- \(b\) is the bias term (if `fit_intercept=True`).

### Loss Functions
GSBR supports the following loss functions:

- **Mean Squared Error (MSE)**:
  \[
  L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
  \]

- **Root Mean Squared Error (RMSE)**:
  \[
  L_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
  \]

- **Mean Absolute Error (MAE)**:
  \[
  L_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
  \]

### Regularization
To mitigate overfitting, regularization terms are added to the loss:

- **L1 (Lasso)**: \(\alpha \sum |w_i|\)
- **L2 (Ridge)**: \(\alpha \sum w_i^2\)
- **ElasticNet**: \(\alpha \left[ l1\_ratio \cdot \sum |w_i| + (1 - l1\_ratio) \cdot \sum w_i^2 \right]\)

The total loss is the base loss plus the regularization penalty.

### Gradients
Optimization uses gradient descent. The gradients for weights and bias are computed as follows (for MSE as an example; similar derivations apply to RMSE and MAE):

- **Weight Gradient**:
  \[
  \frac{\partial L}{\partial w} = \frac{2}{N} X^T (Xw + b - y) + \text{regularization gradient}
  \]

- **Bias Gradient** (if `fit_intercept=True`):
  \[
  \frac{\partial L}{\partial b} = \frac{2}{N} \sum (Xw + b - y)
  \]

For MAE, gradients use the sign function for robustness.

## Key Features
- ✅ Supports L1, L2, and ElasticNet regularization (or none)
- ✅ Loss functions: MSE, RMSE, MAE
- ✅ Handles dense and sparse matrices (via SciPy)
- ✅ Gradient descent with learning rate control
- ✅ Early stopping based on loss convergence
- ✅ Data shuffling for stochastic behavior
- ✅ Verbose logging with customizable levels
- ✅ Random seed for reproducibility
- ✅ Comprehensive input validation and error handling for NaN/Inf values

## Parameters

| Parameter       | Type                          | Default    | Description |
|-----------------|-------------------------------|------------|-------------|
| `max_iter`     | `int`                        | `1000`    | Maximum number of gradient descent iterations. |
| `learning_rate`| `float`                      | `0.01`    | Step size for gradient descent updates. |
| `penalty`      | `Literal['l1', 'l2', 'elasticnet'] \| None` | `'l2'`   | Type of regularization (or `None` for no regularization). |
| `alpha`        | `float`                      | `0.0001`  | Regularization strength. |
| `l1_ratio`     | `float`                      | `0.5`     | Mixing parameter for ElasticNet (0 = L2, 1 = L1). |
| `loss`         | `Literal['mse', 'rmse', 'mae']` | `'mse'`  | Loss function to minimize. |
| `fit_intercept`| `bool`                       | `True`    | Whether to include a bias term (intercept). |
| `tol`          | `float`                      | `0.0001`  | Tolerance for early stopping based on loss convergence. |
| `shuffle`      | `bool`                       | `True`    | Whether to shuffle data before each epoch. |
| `random_state` | `int \| None`                | `None`    | Seed for random number generator (for shuffling and reproducibility). |
| `early_stopping`| `bool`                      | `True`    | Enable early stopping if loss stops improving. |
| `verbose`      | `int`                        | `0`       | Verbosity level: 0 (silent), 1 (progress every ~5% epochs), 2 (every epoch). |

## Model Attributes (After Fitting)

| Attribute      | Type          | Description |
|----------------|---------------|-------------|
| `weights`     | `np.ndarray` | Learned feature weights. |
| `b`           | `float`      | Bias term (intercept). |
| `loss_history`| `List[float]`| Loss values recorded at each iteration. |

## API Reference

### `BasicRegressor.__init__(...)`
Initializes the regressor with the specified hyperparameters.

### `BasicRegressor.fit(X_train, y_train)`
Trains the model on the provided training data using gradient descent.

- **Parameters**:
  - `X_train`: `np.ndarray` or `spmatrix` – Training features (n_samples, n_features).
  - `y_train`: `np.ndarray` or `spmatrix` – Training target values (n_samples,).

- **Raises**:
  - `ValueError`: For invalid inputs (e.g., NaN/Inf, mismatched shapes).
  - `OverflowError`: If parameters become unstable (NaN/Inf) during training.

### `BasicRegressor.predict(X_test)`
Generates predictions for new data.

- **Parameters**:
  - `X_test`: `np.ndarray` – Test features (n_samples, n_features).

- **Returns**:
  - `np.ndarray`: Predicted values (n_samples,).

- **Raises**:
  - `ValueError`: If the model has not been fitted.

## Usage Examples

### Basic Regression with L2 Regularization and MSE Loss
```python
import numpy as np
from sklearn.datasets import make_regression
from BasicModels.GSBR import BasicRegressor

# Generate sample data
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Initialize and train
model = BasicRegressor(max_iter=1000, learning_rate=0.01, penalty='l2', alpha=0.1, verbose=1)
model.fit(X, y)

# Predict and inspect
preds = model.predict(X)
print(f"Final weights: {model.weights}")
print(f"Final bias: {model.b}")
print(f"Final loss: {model.loss_history[-1]:.6f}")
```

### ElasticNet with MAE Loss and Shuffling
```python
from sklearn.preprocessing import StandardScaler
from BasicModels.GSBR import BasicRegressor

# Scale data for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = BasicRegressor(
    max_iter=2000, learning_rate=0.005, penalty='elasticnet', alpha=0.05, l1_ratio=0.3,
    loss='mae', shuffle=True, random_state=42, verbose=2
)
model.fit(X_scaled, y)

print(f"Final MAE loss: {model.loss_history[-1]:.4f}")
```

### Sparse Matrix Support with No Regularization
```python
from scipy.sparse import csr_matrix
from BasicModels.GSBR import BasicRegressor

# Create sparse data
X_sparse = csr_matrix(np.random.randn(100, 10))
y = np.random.randn(100)

model = BasicRegressor(max_iter=500, learning_rate=0.02, penalty=None, verbose=1)
model.fit(X_sparse, y)

print("Sparse training completed!")
```

## Best Practices

### Hyperparameter Tuning
- Start with `learning_rate` between 0.001–0.1; monitor convergence via `loss_history`.
- Use `max_iter=1000–5000` for most datasets; enable `early_stopping` to avoid unnecessary computations.
- For regularization, begin with small `alpha` (e.g., 0.001) and tune `l1_ratio` for ElasticNet.
- Set `shuffle=True` for better generalization on non-i.i.d. data.

### Data Preprocessing
- Normalize/scale features (e.g., using `StandardScaler`) to improve convergence speed.
- Handle missing values and outliers before fitting.
- For sparse data, use SciPy's `csr_matrix` or `csc_matrix` for efficiency.

### Monitoring Training
Visualize loss convergence to diagnose issues:

```python
import matplotlib.pyplot as plt

# After fitting
plt.plot(model.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()
```

## Error Handling
- **Input Validation**: Checks for NaN/Inf in data and shape mismatches.
- **Training Stability**: Monitors for overflow (NaN/Inf in weights/bias) and stops early if detected.
- **Parameter Checks**: Ensures valid `penalty`, `loss`, and range-bound values (e.g., `0 <= l1_ratio <= 1`).

## Performance Considerations
- **Efficiency**: Batch gradient descent works well for small-to-medium datasets (up to ~10k samples).
- **Memory**: Supports sparse matrices to handle high-dimensional data.
- **Scalability**: For larger datasets, consider mini-batch extensions or alternative optimizers.

## Comparison with scikit-learn

| Feature              | GSBR                  | scikit-learn (LinearRegression/ElasticNet) |
|----------------------|-----------------------|--------------------------------------------|
| Regularization       | ✅ (L1/L2/ElasticNet) | ✅                                          |
| Sparse Support       | ✅                    | ✅                                          |
| Gradient Descent     | ✅                    | ❌ (Analytical solvers)                    |
| Multiple Loss Functions | ✅ (MSE/RMSE/MAE)   | ❌ (Primarily MSE)                         |
| Early Stopping       | ✅                    | ✅ (Limited)                                |
| Shuffling & Random Seed | ✅                 | ❌                                          |
| Verbose Logging      | ✅ (Multi-level)      | ✅ (Basic)                                  |
| Cross-Validation     | ❌                    | ✅                                          |

## License
This model is provided as part of the `BasicModels` package under an open-source license (e.g., MIT). Refer to the package repository for details.