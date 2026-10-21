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
    y, tx = _validate_regression_inputs(y, tx)
    w = np.asarray(initial_w, dtype=np.float64).copy()
    if w.ndim != 1 or w.size != tx.shape[1]:
        raise ValueError("initial_w must be a one-dimensional array of length D")
    if not np.all(np.isfinite(w)):
        raise ValueError("initial_w must contain only finite values")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 0:
        raise ValueError("max_iters must be a non-negative integer")
    if not np.isscalar(gamma) or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite non-negative scalar")

    num_samples = y.size
    for _ in range(max_iters):
        residuals = tx @ w - y
        gradient = tx.T @ residuals / num_samples
        w = w - gamma * gradient

    loss = _compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Train linear regression with stochastic gradient descent."""
    y, tx = _validate_regression_inputs(y, tx)
    w = np.asarray(initial_w, dtype=np.float64).copy()
    if w.ndim != 1 or w.size != tx.shape[1]:
        raise ValueError("initial_w must be a one-dimensional array of length D")
    if not np.all(np.isfinite(w)):
        raise ValueError("initial_w must contain only finite values")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 0:
        raise ValueError("max_iters must be a non-negative integer")
    if not np.isscalar(gamma) or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite non-negative scalar")

    num_samples = y.size
    for _ in range(max_iters):
        sample_index = np.random.randint(num_samples)
        residual = tx[sample_index] @ w - y[sample_index]
        gradient = residual * tx[sample_index]
        w = w - gamma * gradient

    loss = _compute_mse(y, tx, w)
    return w, loss


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


#-----------------------------------------------------------------------------------------------
# logistic regression, reg_logistic regression
#-----------------------------------------------------------------------------------------------
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ 
    input:
        y: (N,) label in {0, 1}; tx: (N, D)
        initial_w: (D,); max_iters: int; gamma: float
    out:
        w: (D,); loss: float = sum(log Prob)
    """
    # validate
    y, tx = _validate_logistic_inputs(y, tx)
    
    w = np.asarray(initial_w, dtype=np.float64).copy()
    if w.ndim != 1 or w.size != tx.shape[1]:
        raise ValueError("initial_w must be a one-dimensional array of length D")
    
    # calculate loss
    num_samples = y.size
    for _ in range(max_iters):
        z = tx @ w
        prediction_01 = _sigmoid(z) # in (0,1)
        gradient = tx.T @ (prediction_01 - y) / num_samples
        w = w - gamma * gradient

    loss = _compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    input:
        y: (N,) label in {0, 1}; tx: (N, D)
        initial_w: weight (D,); max_iters: int; gamma: float
        lambda_: float, regularization para, penalty for excessive weights and fit the data with smaller weights.
    out:
        w: (D,); loss: float
    """
    # validate 
    y, tx = _validate_logistic_inputs(y, tx)

    w = np.asarray(initial_w, dtype=np.float64).copy()
    if w.ndim != 1 or w.size != tx.shape[1]:
        raise ValueError("initial_w must be a one-dimensional array of length D")
    
    if not np.isscalar(lambda_) or not np.isfinite(lambda_):
        raise ValueError("lambda_ must be a finite scalar")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    
    # calculate loss
    num_samples = y.size
    for _ in range(max_iters):
        z = tx @ w
        prediction_01 = _sigmoid(z)
        gradient_reg = tx.T @ (prediction_01 - y) / num_samples + 2.0 * lambda_ * w
        w = w - gamma * gradient_reg
        
    loss = _compute_logistic_loss(y, tx, w) # not include the penalty term 2.0 * lambda_ * w !!
    return w, loss


def _validate_logistic_inputs(y, tx):
    """
    check dim, nonempty
    check y in {0,1} instead of {-1,1}
    """
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
    if not np.all(np.isfinite(tx)):
        raise ValueError("tx must contain only finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 and 1 for logistic regression")
    return y, tx

    
def _sigmoid(z):
    """
    Uses two branches to avoid overflow in exp:    
    - z >= 0: 1 / (1 + exp(-z))
    - z <  0: exp(z) / (1 + exp(z))
    """
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)
    positive = z >= 0 # we can also use np.logaddexp(0.0, z)
    negative = ~positive
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    result[negative] = np.exp(z[negative]) / (1.0 + np.exp(z[negative]))
    return result

    
def _compute_logistic_loss(y, tx, w):
    z = tx @ w
    log_term = np.logaddexp(0.0, z) # np.log(1 + np.exp(z)) overflow 
    loss_per_sample = log_term - y * z
    return np.mean(loss_per_sample)


#-----------------------------------------------------------------------------------------------
# linear support vector machine
#-----------------------------------------------------------------------------------------------
def linear_svm_sgd(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    gamma,
    seed=42,
    positive_class_weight=1.0,
):
    """Train a linear soft-margin SVM with stochastic subgradient descent.

    The optimized primal objective is

        mean(max(0, 1 - y * (tx @ w))) + lambda_ / 2 * ||w||^2.

    Labels must use the SVM convention ``{-1, +1}``. At every iteration one
    training sample is drawn uniformly and its hinge-loss subgradient is used.
    The learning rate is kept fixed at ``gamma`` so experiments can control it
    explicitly. Pass an intercept column in ``tx`` if an intercept is desired.

    Args:
        y: One-dimensional labels in ``{-1, +1}`` with shape ``(N,)``.
        tx: Feature matrix with shape ``(N, D)``.
        lambda_: Non-negative L2 regularization strength.
        initial_w: Initial weights with shape ``(D,)``.
        max_iters: Number of stochastic updates.
        gamma: Non-negative learning rate.
        seed: Seed for the local random-number generator.
        positive_class_weight: Multiplier applied to positive-class hinge loss.
            Values greater than one can compensate for class imbalance.

    Returns:
        A tuple ``(w, loss)`` containing the learned weights and the final
        regularized primal objective.
    """
    y, tx = _validate_svm_inputs(y, tx)
    w = np.asarray(initial_w, dtype=np.float64).copy()
    if w.ndim != 1 or w.size != tx.shape[1]:
        raise ValueError("initial_w must be a one-dimensional array of length D")
    if not np.all(np.isfinite(w)):
        raise ValueError("initial_w must contain only finite values")
    if not np.isscalar(lambda_) or not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be a finite non-negative scalar")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 0:
        raise ValueError("max_iters must be a non-negative integer")
    if not np.isscalar(gamma) or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite non-negative scalar")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        not np.isscalar(positive_class_weight)
        or not np.isfinite(positive_class_weight)
        or positive_class_weight <= 0
    ):
        raise ValueError("positive_class_weight must be a finite positive scalar")

    rng = np.random.default_rng(seed)
    num_samples = y.size

    for _ in range(max_iters):
        sample_index = rng.integers(num_samples)
        sample_x = tx[sample_index]
        sample_y = y[sample_index]
        margin = sample_y * (sample_x @ w)
        sample_weight = positive_class_weight if sample_y == 1 else 1.0

        if margin < 1.0:
            subgradient = lambda_ * w - sample_weight * sample_y * sample_x
        else:
            subgradient = lambda_ * w

        w = w - gamma * subgradient

    loss = _compute_svm_objective(
        y, tx, w, lambda_, positive_class_weight=positive_class_weight
    )
    return w, loss


def predict_linear_svm(tx, w):
    """Predict ``{-1, +1}`` labels with a trained linear SVM."""
    tx = np.asarray(tx, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if tx.ndim != 2:
        raise ValueError("tx must be a two-dimensional array")
    if w.ndim != 1 or w.size != tx.shape[1]:
        raise ValueError("w must be a one-dimensional array of length D")
    if not np.all(np.isfinite(tx)) or not np.all(np.isfinite(w)):
        raise ValueError("tx and w must contain only finite values")

    scores = tx @ w
    return np.where(scores >= 0.0, 1, -1)


def _validate_svm_inputs(y, tx):
    """Return validated arrays for a binary SVM using ``{-1, +1}`` labels."""
    y = np.asarray(y, dtype=np.float64)
    tx = np.asarray(tx)
    if not np.issubdtype(tx.dtype, np.number):
        tx = np.asarray(tx, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if tx.ndim != 2:
        raise ValueError("tx must be a two-dimensional array")
    if tx.shape[0] != y.size:
        raise ValueError("tx and y must contain the same number of samples")
    if y.size == 0 or tx.shape[1] == 0:
        raise ValueError("y and tx must not be empty")
    if not np.all(np.isfinite(tx)):
        raise ValueError("tx must contain only finite values")
    if not np.all((y == -1) | (y == 1)):
        raise ValueError("y must contain only -1 and 1 for linear SVM")
    return y, tx


def _compute_svm_objective(y, tx, w, lambda_, positive_class_weight=1.0):
    """Compute mean hinge loss plus the L2 regularization penalty."""
    margins = y * (tx @ w)
    hinge_loss = np.maximum(0.0, 1.0 - margins)
    sample_weights = np.where(y == 1, positive_class_weight, 1.0)
    return np.mean(sample_weights * hinge_loss) + 0.5 * lambda_ * np.dot(w, w)
