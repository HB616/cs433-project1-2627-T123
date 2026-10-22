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


#-----------------------------------------------------------------------------------------------
# binary classification random forest
#-----------------------------------------------------------------------------------------------
DEFAULT_MAX_THRESHOLDS = 64

def random_forest_classifier(y, tx, criterion,
                             num_trees=50, max_tree_depth=10, min_samples_split=20,
                             max_features_mode="sqrt", seed=42,
                             class_balance=None):
    """
    input:
        y: (N,) label in {0, 1}; tx: (N, D) feature matrix
        min_samples_split: int, minimum samples to allow a split
        max_features_mode: num of features per split. max_features = int(sqrt(D)) in "sqrt" mode, int(log2(D)) in "log" mode
    out:
        model: dict containing the list of trees and hyperparameters
        train_loss: float, negative mean log-likelihood on training set
    steps:
        1. verify: y ∈ {0,1}, tx is 2D
        2. decide max_feature, other para
        3. for every tree:
            a. bootstrap with replacement
            b. choose max_features features from D as initial candidates
            c. call "_build_tree"
        4. construct the model = {"trees": trees, "criterion": criterion, ...}
        5. calculate performance in the training center
    """
    y, tx, criterion = _validate_randomtree_inputs(y, tx, criterion)
    if not isinstance(num_trees, (int, np.integer)) or num_trees <= 0:
        raise ValueError("num_trees must be a positive integer")
    if not isinstance(max_tree_depth, (int, np.integer)) or max_tree_depth <= 0:
        raise ValueError("max_tree_depth must be a positive integer")
    if not isinstance(min_samples_split, (int, np.integer)) or min_samples_split < 2:
        raise ValueError("min_samples_split must be an integer >= 2")
    if class_balance is not None:
        if not np.isscalar(class_balance) or not np.isfinite(class_balance):
            raise ValueError("class_balance must be None or a finite scalar")
        if class_balance < 1.0:
            raise ValueError("class_balance must be >= 1 (e.g. 9, 10, 11); "
                            "use None to disable balancing")

    num_samples, num_features = tx.shape

    if max_features_mode == "sqrt":
        n_features_per_split = max(1, int(np.sqrt(num_features)))
    elif max_features_mode == "log":
        n_features_per_split = max(1, int(np.log2(num_features)))
    elif isinstance(max_features_mode, (int, np.integer)):
        if max_features_mode <= 0:
            raise ValueError("max_features_mode as int must be positive")
        n_features_per_split = int(max_features_mode)
    else:
        raise ValueError("max_features_mode must be 'sqrt', 'log' or a positive int")

    rng = np.random.default_rng(seed)
    trees = []
    for _ in range(num_trees):
        indices = rng.integers(0, num_samples, size=num_samples)
        x_boot = tx[indices]
        y_boot = y[indices]
        if class_balance is not None:
            y_boot, x_boot = _balance_classes(y_boot, x_boot, class_balance, rng)
        tree = _build_tree(
            x_boot, y_boot,
            depth=0,
            max_depth=max_tree_depth,
            min_samples_split=min_samples_split,
            n_features=n_features_per_split,
            rng=rng,
            criterion=criterion,
        )
        # flatten the tree once for fast prediction
        tree["_flat"] = _flatten_tree(tree)
        trees.append(tree)

    model = {
        "trees": trees,
        "criterion": criterion,
        "num_trees": num_trees,
        "max_tree_depth": max_tree_depth,
        "min_samples_split": min_samples_split,
        "n_features_per_split": n_features_per_split,
        "seed": seed,
        "class_balance": class_balance,
    }

    prob = random_forest_predict_proba(model, tx)
    eps = 1e-12
    prob = np.clip(prob, eps, 1.0 - eps)
    train_loss = -np.mean(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob))
    return model, train_loss


def _validate_randomtree_inputs(y, tx, criterion):
    """
    check dim, nonempty
    check y in {0,1} instead of {-1,1}
    """
    y = np.asarray(y, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64)
    if criterion not in ("gini", "entropy"):
        raise ValueError("criterion must be 'gini' or 'entropy'")
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
        raise ValueError("y must contain only 0 and 1 for classification")
    return y, tx, criterion


def _balance_classes(y, tx, pos_multiplier, rng):
    """
    Oversample the positive class
    input:
        pos_multiplier: float >= 1, e.g. 9, 10, 11
        rng: numpy Generator used for reproducible resampling
    out:
        y_bal: (N',) ; tx_bal: (N', D) with N' = pos_multiplier * N_pos + N_neg
    """
    pos_idx = np.flatnonzero(y == 1.0)
    neg_idx = np.flatnonzero(y == 0.0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise ValueError("both classes must be present to apply class balancing")

    n_pos_new = int(round(pos_multiplier * pos_idx.size))
    new_pos = rng.choice(pos_idx, size=n_pos_new, replace=True)    # 正类有放回复制
    idx = np.concatenate([new_pos, neg_idx])                       
    rng.shuffle(idx)                                               # 打乱，避免树内顺序偏置
    return y[idx], tx[idx]


def random_forest_predict_proba(model, tx):
    """
    For each sample x:
        1. traverse all trees in the forest and obtain the probability of class 1
        2. take the average of the probabilities given by these trees
    """
    tx = np.asarray(tx, dtype=np.float64)
    if tx.ndim != 2:
        raise ValueError("tx must be a two-dimensional array")
    num_samples = tx.shape[0]
    probs = np.zeros(num_samples, dtype=np.float64)
    for tree in model["trees"]:
        probs += _predict_tree_flat(tree["_flat"], tx)
    return probs / len(model["trees"])


#-----------------------------------------------
def _impurity(y, criterion):
    if criterion == "gini":
        return _gini_impurity(y)
    elif criterion == "entropy":
        return _entropy(y)
    else:
        raise ValueError("criterion must be 'gini' or 'entropy'")


def _entropy(y):
    """
    Shannon entropy for binary labels in {0, 1}:
        H = - p1 * log2(p1) - p2 * log2(p2)
    """
    p1 = np.clip(np.mean(y), 1e-12, 1.0 - 1e-12)
    p2 = 1.0 - p1
    return -(p1 * np.log2(p1) + p2 * np.log2(p2))


def _gini_impurity(y):
    p1 = np.mean(y)
    p2 = 1.0 - p1
    return 1.0 - p1 * p1 - p2 * p2


def _best_split(x, y, feature_indices, rng, criterion, max_thresholds=DEFAULT_MAX_THRESHOLDS):
    """
    For each candidate feature and threshold, compute resulting gain:
        parent_impurity = _impurity(y, criterion)
        weighted_impurity = (n_left/n) * _impurity(y_left, criterion)
                          + (n_right/n) * _impurity(y_right, criterion)
        gain = parent_impurity - weighted_impurity
    Return (feature, threshold, gain) with the largest gain,
    or (None, None, 0) if no valid split is found.

    Vectorized implementation: the per-threshold loop is replaced by
    NumPy array operations over all candidate positions at once.
    """
    n = y.size
    parent_impurity = _impurity(y, criterion)
    best_gain = 0.0
    best_feature = None
    best_threshold = None

    for j in feature_indices:
        column = x[:, j]
        order = np.argsort(column, kind="mergesort")
        sorted_col = column[order]
        sorted_y = y[order]

        cum_pos = np.cumsum(sorted_y)
        total_pos = int(cum_pos[-1])

        # candidate split positions: only where the value changes
        change = np.flatnonzero(sorted_col[1:] != sorted_col[:-1]) + 1
        if change.size == 0:
            continue

        # subsample candidate positions if too many
        if change.size > max_thresholds:
            idx = rng.choice(change.size, size=max_thresholds, replace=False)
            positions = np.sort(change[idx])
        else:
            positions = change

        # filter invalid positions (must split into non-empty left and right)
        valid = (positions > 0) & (positions < n)
        positions = positions[valid]
        if positions.size == 0:
            continue

        # vectorized counts
        n_left = positions.astype(np.float64)
        n_right = n - n_left
        n_pos_left = cum_pos[positions - 1].astype(np.float64)
        n_pos_right = total_pos - n_pos_left

        p_left = n_pos_left / n_left
        p_right = n_pos_right / n_right

        # vectorized impurity
        if criterion == "gini":
            imp_left = 2.0 * p_left * (1.0 - p_left)
            imp_right = 2.0 * p_right * (1.0 - p_right)
        else:  # entropy
            p_left_clip = np.clip(p_left, 1e-12, 1.0 - 1e-12)
            p_right_clip = np.clip(p_right, 1e-12, 1.0 - 1e-12)
            imp_left = -(p_left_clip * np.log2(p_left_clip)
                         + (1.0 - p_left_clip) * np.log2(1.0 - p_left_clip))
            imp_right = -(p_right_clip * np.log2(p_right_clip)
                          + (1.0 - p_right_clip) * np.log2(1.0 - p_right_clip))

        weighted = (n_left * imp_left + n_right * imp_right) / n
        gains = parent_impurity - weighted

        # best candidate for this feature
        k = int(np.argmax(gains))
        if gains[k] > best_gain:
            best_gain = float(gains[k])
            best_feature = j
            pos = int(positions[k])
            best_threshold = 0.5 * (sorted_col[pos - 1] + sorted_col[pos])

    return best_feature, best_threshold, best_gain


# Recursively build a decision tree.
def _build_tree(x, y, depth, max_depth, min_samples_split, n_features, rng, criterion):
    """
    Stops when:
        - depth >= max_depth
        - number of samples at current node < min_samples_split
        - y is pure (all labels identical)

    Return a dict node:
        leaf node:     {"leaf": True,  "prob": p1}
        internal node: {"leaf": False, "feature": j, "threshold": t,
                        "left": ..., "right": ...}
    """
    if (depth >= max_depth
            or y.size < min_samples_split
            or y.min() == y.max()):
        return {"leaf": True, "prob": float(np.mean(y))}

    num_features = x.shape[1]
    n_select = min(n_features, num_features)
    feature_indices = rng.choice(num_features, size=n_select, replace=False)

    j, t, gain = _best_split(x, y, feature_indices, rng, criterion)
    if j is None or gain <= 0.0:
        return {"leaf": True, "prob": float(np.mean(y))}

    left_mask = x[:, j] <= t
    left = _build_tree(x[left_mask], y[left_mask],
                       depth + 1, max_depth, min_samples_split,
                       n_features, rng, criterion)
    right = _build_tree(x[~left_mask], y[~left_mask],
                        depth + 1, max_depth, min_samples_split,
                        n_features, rng, criterion)
    return {"leaf": False, "feature": int(j), "threshold": float(t),
            "left": left, "right": right}


def _predict_tree_flat(flat, tx):
    """
    Batched traversal of one tree over all rows of tx.
    Returns an array of shape (num_samples,) with P(y=1).
    """
    num_samples = tx.shape[0]
    node_ids = np.zeros(num_samples, dtype=np.int64)    # start at root
    active = np.arange(num_samples, dtype=np.int64)     # still-to-be-evaluated

    while active.size > 0:
        current_nodes = node_ids[active]
        leaf_mask = flat["is_leaf"][current_nodes].astype(bool)
        if np.all(leaf_mask):
            break
        active = active[~leaf_mask]
        if active.size == 0:
            break
        current_nodes = node_ids[active]
        feat = flat["feature"][current_nodes]
        thr = flat["threshold"][current_nodes]
        values = tx[active, feat]
        go_left = values <= thr
        node_ids[active[go_left]] = flat["left"][current_nodes[go_left]]
        node_ids[active[~go_left]] = flat["right"][current_nodes[~go_left]]

    return flat["prob"][node_ids]


# Flat-array representation for fast batched prediction
def _flatten_tree(root):
    """
    Convert a nested tree (dict) into flat numpy arrays:
        feature[i], threshold[i]  (for internal nodes; -1 for leaves)
        left[i], right[i]
        is_leaf[i]
        prob[i]
    Node 0 is the root.
    """
    features = []
    thresholds = []
    left = []
    right = []
    is_leaf = []
    prob = []

    def visit(node):
        idx = len(features)
        features.append(-1)
        thresholds.append(0.0)
        left.append(-1)
        right.append(-1)
        is_leaf.append(0)
        prob.append(0.0)

        if node["leaf"]:
            is_leaf[idx] = 1
            prob[idx] = float(node["prob"])
            return idx

        features[idx] = int(node["feature"])
        thresholds[idx] = float(node["threshold"])
        left_child = visit(node["left"])
        right_child = visit(node["right"])
        left[idx] = left_child
        right[idx] = right_child
        return idx

    visit(root)
    return {
        "feature": np.asarray(features, dtype=np.int64),
        "threshold": np.asarray(thresholds, dtype=np.float64),
        "left": np.asarray(left, dtype=np.int64),
        "right": np.asarray(right, dtype=np.int64),
        "is_leaf": np.asarray(is_leaf, dtype=np.int8),
        "prob": np.asarray(prob, dtype=np.float64),
    }