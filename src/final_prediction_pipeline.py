import os
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def load_handcrafted_rt_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the RT activity dataset backed by handcrafted features only,
    applying the same merge and leakage-prevention steps used in the
    Optuna and feature-importance scripts.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with FoldSeek-derived columns removed.
    y : pd.Series
        Binary activity labels.
    meta : pd.DataFrame
        Metadata columns (at least rt_name, active).
    """
    features_df = pd.read_csv("data/handcrafted_features.csv")
    sequences_df = pd.read_csv("data/rt_sequences.csv")

    df = features_df.merge(sequences_df[["rt_name", "active"]], on="rt_name", how="inner")

    X = df.drop(["rt_name", "active"], axis=1)
    y = df["active"]

    # Remove FoldSeek columns to match the leak-free training setup
    foldseek_cols = [c for c in X.columns if "foldseek" in c]
    if foldseek_cols:
        X = X.drop(columns=foldseek_cols)

    meta = df[["rt_name", "active"]].copy()
    return X, y, meta


def get_best_xgb_model(random_state: int = 42) -> xgb.XGBClassifier:
    """
    Instantiate the final XGBoost model using the best parameters
    discovered via Optuna (≈80.7–80.8% accuracy in 5-fold CV).
    """
    best_params: Dict[str, Any] = {
        "n_estimators": 188,
        "max_depth": 4,
        "learning_rate": 0.05474919881626068,
        "subsample": 0.7167183980497441,
        "colsample_bytree": 0.5895901388066157,
        "gamma": 3.0690014078763155,
        "scale_pos_weight": 2.032711735305366,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        "use_label_encoder": False,
    }
    return xgb.XGBClassifier(**best_params)


def cross_validated_report(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run Stratified K-Fold cross validation with the final model and
    return metrics plus per-sample predictions.
    """
    model = get_best_xgb_model(random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Cross-validated predictions to approximate held-out performance
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": y.to_numpy(),
        "y_pred": y_pred,
    }


def train_final_model(
    save_dir: str | Path = "models",
    model_name: str = "final_xgb_optuna.pkl",
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """
    Train the final XGBoost model on the full leak-free dataset and
    optionally persist it to disk for downstream inference.

    Returns
    -------
    model : xgb.XGBClassifier
        Fitted model ready for prediction.
    """
    X, y, _ = load_handcrafted_rt_data()
    model = get_best_xgb_model(random_state=random_state)
    model.fit(X, y)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / model_name

    joblib.dump(model, model_path)
    print(f"Saved final model to: {model_path}")

    return model


def load_final_model(model_path: str | Path = "models/final_xgb_optuna.pkl") -> xgb.XGBClassifier:
    """
    Load a previously trained final model from disk.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            f"Train it first by running this module as a script."
        )
    return joblib.load(model_path)


def predict_from_dataframe(
    df: pd.DataFrame,
    model: xgb.XGBClassifier | None = None,
) -> np.ndarray:
    """
    Predict RT activity from a dataframe that matches the handcrafted
    features schema (including rt_name/active if present).
    """
    if model is None:
        model = load_final_model()

    X = df.copy()
    # Drop label/identifier columns if present
    for col in ("rt_name", "active", "rt_family"):
        if col in X.columns:
            X = X.drop(columns=[col])

    # Ensure FoldSeek leakage columns are removed here as well
    foldseek_cols = [c for c in X.columns if "foldseek" in c]
    if foldseek_cols:
        X = X.drop(columns=foldseek_cols)

    return model.predict(X)


if __name__ == "__main__":
    print("🔁 Building final RT activity prediction pipeline with Optuna-tuned XGBoost...")

    # 1. Load data using the same preprocessing as the Optuna study
    X, y, meta = load_handcrafted_rt_data()
    print(f"Loaded {len(X)} sequences with {X.shape[1]} features (FoldSeek removed).")

    # 2. Cross-validated performance (should be ~0.807–0.808 accuracy)
    results = cross_validated_report(X, y, n_splits=5, random_state=42)
    acc = results["accuracy"]
    cm = results["confusion_matrix"]

    print(f"\nCross-validated accuracy (Stratified 5-fold): {acc:.4f}")
    print("\nConfusion matrix (CV predictions):")
    print(cm)

    # 3. Fit on full data and persist the final model
    model = train_final_model()

    print("\nFinal model is trained on the full dataset and saved for deployment.")

