"""
breach_classifier.py - Supervised machine learning module for Value at Risk (VaR) breach prediction.

Builds backward-looking predictive features (rolling volatility, momentum, volume trend),
generates time-series labels for historical VaR breaches, and evaluates classification models
(Random Forest and Logistic Regression) with chronological train/test splits.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def build_breach_features(
    portfolio_returns: pd.Series,
    volume_df: Optional[pd.DataFrame] = None,
    weights: Optional[Union[list, np.ndarray]] = None,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Constructs backward-looking features (rolling vol, 5d momentum, volume trend) strictly without look-ahead bias."""
    # 1. Realized 20-day annualized volatility from past returns (t-1 backwards)
    rolling_vol = portfolio_returns.shift(1).rolling(window=20).std() * np.sqrt(trading_days)

    # 2. 5-day trailing compounded momentum (t-1 backwards)
    trailing_5d_mom = (
        portfolio_returns.shift(1)
        .rolling(window=5)
        .apply(lambda r: float((1.0 + r).prod() - 1.0), raw=False)
    )

    # 3. Rolling 20-day volume trend (% change vs 20-day moving average of volume)
    if volume_df is not None and weights is not None and not volume_df.empty:
        weights_arr = np.array(weights, dtype=float)
        port_volume = volume_df.dot(weights_arr)
        prior_volume = port_volume.shift(1)
        avg_volume_20d = prior_volume.rolling(window=20).mean()
        volume_trend = (prior_volume - avg_volume_20d) / avg_volume_20d.replace(0, np.nan)
        volume_trend = volume_trend.fillna(0.0)
    else:
        # Fallback to zero trend if volume data is unavailable
        volume_trend = pd.Series(0.0, index=portfolio_returns.index)

    features_df = pd.DataFrame(
        {
            "Rolling 20D Volatility": rolling_vol,
            "5D Momentum": trailing_5d_mom,
            "20D Volume Trend": volume_trend,
        },
        index=portfolio_returns.index,
    )

    return features_df


def generate_breach_labels(
    portfolio_returns: pd.Series, confidence_level: float = 0.95
) -> pd.Series:
    """Generates binary labels: 1 if daily return is at or below the historical VaR threshold, 0 otherwise."""
    percentile_cutoff = (1.0 - confidence_level) * 100.0
    var_cutoff = float(np.percentile(portfolio_returns, percentile_cutoff))
    labels = (portfolio_returns <= var_cutoff).astype(int)
    labels.name = "VaR_Breach"
    return labels


def time_series_train_test_split(
    X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits feature matrix X and label series y chronologically (first 80% train, last 20% test)."""
    # Align indices and drop initial warm-up NaNs from rolling windows
    aligned_data = pd.concat([X, y], axis=1).dropna()
    clean_X = aligned_data[X.columns]
    clean_y = aligned_data[y.name]

    split_idx = int(len(clean_X) * train_ratio)
    if split_idx < 10 or (len(clean_X) - split_idx) < 5:
        raise ValueError("Insufficient data points for a meaningful time series train/test split.")

    X_train = clean_X.iloc[:split_idx]
    X_test = clean_X.iloc[split_idx:]
    y_train = clean_y.iloc[:split_idx]
    y_test = clean_y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def train_breach_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "Random Forest",
    random_state: int = 42,
) -> Any:
    """Trains a supervised classification model (Random Forest or Logistic Regression) with balanced class weights."""
    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
            class_weight="balanced",
        )
    elif model_type == "Logistic Regression":
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'Random Forest' or 'Logistic Regression'.")

    model.fit(X_train, y_train)
    return model


def evaluate_breach_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """Computes test set classification metrics (precision, recall, F1, accuracy, confusion matrix)."""
    y_pred = model.predict(X_test)

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    accuracy = float((y_test.values == y_pred).mean())
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "y_pred": y_pred,
        "y_test": y_test.values,
        "total_test_samples": len(y_test),
        "test_breaches": int(y_test.sum()),
        "predicted_breaches": int(y_pred.sum()),
    }


def extract_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extracts tree-based feature importance or logistic regression coefficients as a formatted DataFrame."""
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
        importance_type = "Feature Importance (MDI)"
    elif hasattr(model, "coef_"):
        importance_values = model.coef_[0]
        importance_type = "Logistic Coefficient"
    else:
        importance_values = np.zeros(len(feature_names))
        importance_type = "Unknown"

    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_values,
            "Type": importance_type,
        }
    )
    return df.sort_values(by="Importance", ascending=True).reset_index(drop=True)
