import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def load_handcrafted_rt_data():
    """
    Load handcrafted RT features, add Isolectric Point and Molecular
    Weight if available in the sequence CSV, and clean NaN/Inf values.
    """
    features_df = pd.read_csv("data/handcrafted_features.csv")
    sequences_df = pd.read_csv("data/rt_sequences.csv")

    # Add extra physico‑chemical features from the sequences file if
    # they are not already present in the handcrafted feature table.
    extra_cols = []
    for col in ["Isoelectric Point", "Molecular Weight"]:
        if col in sequences_df.columns and col not in features_df.columns:
            extra_cols.append(col)

    seq_cols = ["rt_name", "active"] + extra_cols
    seq_cols = [c for c in seq_cols if c in sequences_df.columns]

    df = features_df.merge(sequences_df[seq_cols], on="rt_name", how="inner")

    X = df.drop(["rt_name", "active"], axis=1)
    y = df["active"]

    # Remove FoldSeek columns (leakage prevention)
    foldseek_cols = [c for c in X.columns if "foldseek" in c]
    if foldseek_cols:
        X = X.drop(columns=foldseek_cols)

    # Replace Inf / -Inf with NaN, then impute with column medians
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def get_best_xgb_model():
    # Optuna Winning Parameters
    best_params = {
        "n_estimators": 188,
        "max_depth": 4,
        "learning_rate": 0.05474919881626068,
        "subsample": 0.7167183980497441,
        "colsample_bytree": 0.5895901388066157,
        "gamma": 3.0690014078763155,
        "scale_pos_weight": 2.032711735305366,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
    }
    return xgb.XGBClassifier(**best_params)


if __name__ == "__main__":
    print("\n--- Building Calibrated 80.7% XGBoost Pipeline ---")

    # 1. Data load + feature augmentation + cleaning
    X, y = load_handcrafted_rt_data()
    print(f"Data Loaded: {len(X)} sequences, {X.shape[1]} features (FoldSeek removed, NaN/Inf cleaned).")

    # 2. Performance check (Cross-validation) using the base XGBoost
    base_model = get_best_xgb_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(base_model, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    print(f"Cross-validated Accuracy (base XGBoost): {acc:.4f}")

    # 3. Calibrate probabilities with CalibratedClassifierCV
    print("Fitting CalibratedClassifierCV (isotonic, cv=5) for probability calibration...")
    calibrated_model = CalibratedClassifierCV(
        base_estimator=get_best_xgb_model(),
        method="isotonic",
        cv=5,
    )
    calibrated_model.fit(X, y)

    # Optional: evaluate calibrated probability quality via cross-validated proba
    y_proba = cross_val_predict(
        calibrated_model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    brier = brier_score_loss(y, y_proba)
    ll = log_loss(y, y_proba)
    print(f"Calibrated probability Brier score: {brier:.4f}")
    print(f"Calibrated probability log loss:   {ll:.4f}")

    # 4. Save the calibrated model for CLS submissions
    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = "models/final_xgb_optuna_calibrated.pkl"
    joblib.dump(calibrated_model, model_path)
    print(f"Calibrated model saved: {model_path}\n")