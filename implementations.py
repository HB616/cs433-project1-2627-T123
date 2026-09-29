"""Required machine-learning method implementations for CS-433 Project 1.

The six public functions in this module must keep the exact signatures required
by the project specification. Implementations will be added and verified during
the algorithm-development stage of the project.
"""

import numpy as np


def _validate_regression_inputs(y, tx):
    """Return validated floating-point arrays for regression methods."""
    y = np.asarray(y, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if tx.ndim != 2:
        raise ValueError("tx must be a two-dimensional array")
    if tx.shape[0] != y.size:
        raise ValueError("tx and y must contain the same number of samples")
    if y.size == 0 or tx.shape[1] == 0:
        raise ValueError("y and tx must not be empty")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(tx)):
        raise ValueError("y and tx must contain only finite values")
    return y, tx


def _compute_mse(y, tx, w):
    """Compute the course MSE convention: ||y - Xw||² / (2N)."""
    residuals = y - tx @ w
    return np.sum(residuals**2) / (2.0 * y.size)


def _solve_linear_system(matrix, vector):
    """Solve a linear system, using the pseudoinverse when it is singular."""
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ vector


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Train linear regression with batch gradient descent."""
    raise NotImplementedError("Implementation pending")


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Train linear regression with stochastic gradient descent."""
    raise NotImplementedError("Implementation pending")


def least_squares(y, tx):
    """Fit least-squares regression using the normal equations.

    A pseudoinverse fallback provides a minimum-norm solution when ``tx.T @ tx``
    is singular. The returned loss follows the course convention with a factor
    of one half.
    """
    y, tx = _validate_regression_inputs(y, tx)
    gram_matrix = tx.T @ tx
    right_hand_side = tx.T @ y
    w = _solve_linear_system(gram_matrix, right_hand_side)
    loss = _compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Fit ridge regression using the regularized normal equations.

    The optimized objective is ``MSE + lambda_ * ||w||²``. As required by the
    project specification, the returned loss is the unregularized MSE and does
    not include the penalty term.
    """
    y, tx = _validate_regression_inputs(y, tx)
    if not np.isscalar(lambda_) or not np.isfinite(lambda_):
        raise ValueError("lambda_ must be a finite scalar")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")

    num_samples, num_features = tx.shape
    regularization = 2.0 * num_samples * lambda_ * np.eye(num_features)
    system_matrix = tx.T @ tx + regularization
    right_hand_side = tx.T @ y
    w = _solve_linear_system(system_matrix, right_hand_side)
    loss = _compute_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Train binary logistic regression with gradient descent."""
    raise NotImplementedError("Implementation pending")


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Train L2-regularized logistic regression with gradient descent."""
    raise NotImplementedError("Implementation pending")
