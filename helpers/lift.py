import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def area_ratio(y_true, y_prob):
    """
    Calculate the Area Ratio from a Cumulative Accuracy Profile (CAP) curve.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 = negative, 1 = positive).
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Area ratio, representing the normalized gain between the random baseline and the perfect model.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    n = len(y_true)
    total_positives = y_true.sum()

    # Sort by predicted probability (descending)
    sorted_idx = np.argsort(-y_prob)
    y_sorted = y_true.iloc[sorted_idx]

    # Cumulative positives
    cum_positives = np.cumsum(y_sorted)
    cap_curve = cum_positives / total_positives

    # Ideal CAP: all positives first, then negatives
    ideal_cap = np.concatenate([
        np.linspace(1 / total_positives, 1, int(total_positives)),
        np.ones(n - int(total_positives))
    ])

    # Random CAP: diagonal baseline
    random_cap = np.linspace(0, 1, n)

    # Compute areas
    area_model = np.trapezoid(cap_curve)
    area_random = np.trapezoid(random_cap)
    area_perfect = np.trapezoid(ideal_cap)

    return (area_model - area_random) / (area_perfect - area_random)


def area_ratio_scorer(estimator, X, y):
    """
    Scikit-learn compatible scorer for Area Ratio.

    Parameters
    ----------
    estimator : object
        Fitted scikit-learn estimator with a predict_proba method.
    X : array-like
        Feature matrix.
    y : array-like
        True binary labels.

    Returns
    -------
    float
        Area ratio score.
    """
    return area_ratio(y, estimator.predict_proba(X)[:, 1])


def plot_knn_area_ratio_curve(df, metric="manhattan", weights="uniform"):
    """
    Plot Area Ratio vs k for KNN models with specified metric and weights.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns: 'k', 'metric', 'weights', 'area_ratio'.
    metric : str, optional
        Distance metric filter (default is 'manhattan').
    weights : str, optional
        KNN weighting scheme (default is 'uniform').
    """
    subset = df[(df["metric"] == metric) & (df["weights"] == weights)].sort_values("k")

    plt.figure(figsize=(8, 6))
    plt.plot(subset["k"], subset["area_ratio"], marker="o", label=f"{metric}, {weights}")
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, label="Plateau ~k=50")

    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Area Ratio (CV)")
    plt.title(f"KNN Area Ratio vs k ({metric} distance, {weights} weights)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.show()


def plot_lift_chart(y_true, y_prob, model_name="Model"):
    """
    Plot a Lift Chart comparing the model, random baseline, and perfect model.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 = negative, 1 = positive).
    y_prob : array-like
        Predicted probabilities for the positive class.
    model_name : str, optional
        Label for the model curve in the plot.
    """
    sorted_idx = np.argsort(-y_prob)
    y_true_sorted = np.array(y_true)[sorted_idx]

    total_defaulters = y_true.sum()
    n = len(y_true)

    # Cumulative defaulters found by the model
    cumulative_defaulters = np.cumsum(y_true_sorted)

    # Random baseline
    baseline = np.linspace(0, total_defaulters, n)

    # Perfect model
    best_curve = np.concatenate([
        np.linspace(0, total_defaulters, int(total_defaulters)),
        np.ones(n - int(total_defaulters)) * total_defaulters
    ])

    x = np.arange(1, n + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, cumulative_defaulters, label=f"{model_name}", color='blue')
    plt.plot(x, baseline, label="Baseline (random)", color='gray', linestyle='--')
    plt.plot(x[:len(best_curve)], best_curve[:len(x)], label="Best model", color='black')

    plt.xlabel("Number of customers (sorted by predicted probability)")
    plt.ylabel("Cumulative number of defaulters captured")
    plt.title(f"Lift Chart - {model_name}")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.show()


def cap_area_ratio(y_true, y_scores):
    """
    Compute the Area Ratio from a CAP curve using predicted scores.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_scores : array-like
        Predicted scores or probabilities for the positive class.

    Returns
    -------
    float
        Area ratio score.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    n = len(y_true)
    total_positives = y_true.sum()

    # Sort by predicted probability (descending)
    sorted_idx = np.argsort(-y_scores)
    y_sorted = y_true.iloc[sorted_idx]

    cum_positives = np.cumsum(y_sorted)
    cap_curve = cum_positives / total_positives

    ideal_cap = np.concatenate([
        np.linspace(1 / total_positives, 1, int(total_positives)),
        np.ones(n - int(total_positives))
    ])

    random_cap = np.linspace(0, 1, n)

    area_model = np.trapezoid(cap_curve)
    area_random = np.trapezoid(random_cap)
    area_perfect = np.trapezoid(ideal_cap)

    return (area_model - area_random) / (area_perfect - area_random)
