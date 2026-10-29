import numpy as np


def _sigmoid(values):
    "Compute the logistic function
    values = np.asarray(values)
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def fit_balanced_logistic_regression(
    y,
    tx,
    epochs=20,
    batch_size=2048,
    learning_rate=0.1,
    regularization=1e-4,
    seed=42,
):
    "Fit L2-regularized logistic regression with majority undersampling
    y = np.asarray(y, dtype=np.float64)
    tx = np.asarray(tx)
    if y.ndim != 1 or tx.ndim != 2 or tx.shape[0] != y.size:
        raise ValueError("y must be 1D and aligned with the rows of 2D tx")
    if y.size == 0 or tx.shape[1] == 0:
        raise ValueError("y and tx must not be empty")
    if not np.all(np.isfinite(y)) or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y must contain only finite 0/1 labels")
    if not np.all(np.isfinite(tx)):
        raise ValueError("tx must contain only finite values")
    if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be non-negative and finite")

    negative_indices = np.flatnonzero(y == 0.0)
    positive_indices = np.flatnonzero(y == 1.0)
    if negative_indices.size == 0 or positive_indices.size == 0:
        raise ValueError("Both classes must be present in the training data")

    rng = np.random.default_rng(seed)
    minority_count = min(negative_indices.size, positive_indices.size)
    if negative_indices.size > minority_count:
        negative_indices = rng.choice(
            negative_indices, size=minority_count, replace=False
        )
    if positive_indices.size > minority_count:
        positive_indices = rng.choice(
            positive_indices, size=minority_count, replace=False
        )

    balanced_indices = np.concatenate((negative_indices, positive_indices))
    weights = np.zeros(tx.shape[1], dtype=np.float64)
    intercept = 0.0

    for epoch in range(epochs):
        rng.shuffle(balanced_indices)
        step_size = learning_rate / np.sqrt(1.0 + epoch / 5.0)
        for start in range(0, balanced_indices.size, batch_size):
            batch_indices = balanced_indices[start : start + batch_size]
            batch_x = tx[batch_indices]
            batch_y = y[batch_indices]
            residual = _sigmoid(batch_x @ weights + intercept) - batch_y
            gradient = batch_x.T @ residual / batch_y.size
            gradient += 2.0 * regularization * weights
            weights -= step_size * gradient
            intercept -= step_size * float(np.mean(residual))

    return weights, float(intercept), (int(minority_count), int(minority_count))


def predict_probabilities(tx, weights, intercept):
    "Return positive-class probabilities for a finite feature matrix
    tx = np.asarray(tx)
    weights = np.asarray(weights, dtype=np.float64)
    if tx.ndim != 2 or weights.ndim != 1 or tx.shape[1] != weights.size:
        raise ValueError("tx columns must match the one-dimensional weights")
    if not np.all(np.isfinite(tx)) or not np.all(np.isfinite(weights)):
        raise ValueError("tx and weights must contain only finite values")
    return _sigmoid(tx @ weights + float(intercept))


def _select_threshold(y_true, probabilities, objective):
    "Select a threshold for one validation metric with deterministic ties
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional")
    if y_true.size == 0 or y_true.shape != probabilities.shape:
        raise ValueError("y_true and probabilities must be non-empty and aligned")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0/1 labels")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")

    if objective not in ("accuracy", "balanced_accuracy", "f1"):
        raise ValueError("objective must be accuracy, balanced_accuracy, or f1")
    positives = y_true == 1
    negatives = ~positives
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negatives))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Validation data must contain both classes")

    if objective == "f1":
        # Evaluate every distinct validation probability as a decision boundary.
        # This avoids a coarse fixed grid missing the F1-optimal threshold.
        thresholds = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
        order = np.argsort(probabilities)
        sorted_probabilities = probabilities[order]
        positive_prefix = np.concatenate(
            ([0], np.cumsum(y_true[order] == 1, dtype=np.int64))
        )
        first_predicted = np.searchsorted(
            sorted_probabilities, thresholds, side="left"
        )
        candidate_true_positives = positive_count - positive_prefix[first_predicted]
        candidate_false_positives = (
            y_true.size - first_predicted - candidate_true_positives
        )
    else:
        thresholds = np.linspace(0.0, 1.0, 201)
    best_key = None
    best_threshold = 0.5
    best_metrics = None

    for candidate_index, threshold in enumerate(thresholds):
        if objective == "f1":
            true_positive = int(candidate_true_positives[candidate_index])
            false_positive = int(candidate_false_positives[candidate_index])
        else:
            predicted = probabilities >= threshold
            true_positive = int(np.sum(predicted & positives))
            false_positive = int(np.sum(predicted & negatives))
        false_negative = positive_count - true_positive
        true_negative = negative_count - false_positive
        recall = true_positive / positive_count
        specificity = true_negative / negative_count
        precision_denominator = true_positive + false_positive
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        f1_denominator = precision + recall
        f1 = 2.0 * precision * recall / f1_denominator if f1_denominator else 0.0
        balanced_accuracy = 0.5 * (recall + specificity)
        accuracy = (true_positive + true_negative) / y_true.size
        if objective == "accuracy":
            key = (
                accuracy,
                balanced_accuracy,
                f1,
                recall,
                -abs(float(threshold) - 0.5),
            )
        elif objective == "balanced_accuracy":
            key = (
                balanced_accuracy,
                f1,
                recall,
                accuracy,
                -abs(float(threshold) - 0.5),
            )
        else:
            key = (
                f1,
                precision,
                recall,
                balanced_accuracy,
                -abs(float(threshold) - 0.5),
            )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1_score": f1,
                "recall": recall,
                "specificity": specificity,
                "precision": precision,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
            }
    return best_threshold, best_metrics


def select_accuracy_threshold(y_true, probabilities):
    "Choose a validation threshold that maximizes plain accuracy.
    return _select_threshold(y_true, probabilities, objective="accuracy")


def select_balanced_accuracy_threshold(y_true, probabilities):
    "Choose a validation threshold that maximizes balanced accuracy
    return _select_threshold(y_true, probabilities, objective="balanced_accuracy")


def select_f1_threshold(y_true, probabilities):
    "Choose the validation decision threshold that maximizes positive-class F1
    return _select_threshold(y_true, probabilities, objective="f1")
