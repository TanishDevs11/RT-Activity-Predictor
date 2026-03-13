import pandas as pd
import joblib
import numpy as np

# 1. Calibrated Model-ah load panrom
model = joblib.load("models/final_xgb_optuna_calibrated.pkl")

# 2. Features-ah load panrom (Make sure this matches training features)
features_df = pd.read_csv("data/handcrafted_features.csv")
X = features_df.drop(["rt_name"], axis=1)
X = X.drop(columns=[c for c in X.columns if "foldseek" in c], errors='ignore')

# 3. Clean NaN/Inf (Same as training)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

# 4. Predict Probabilities (CLS Score uses these!)
# Class 1 (Active) oda probability-ah mattum yedukkurom
probs = model.predict_proba(X)[:, 1]

# 5. Create Submission DataFrame
submission = pd.DataFrame({
    'rt_name': features_df['rt_name'],
    'probability': probs
})

submission.to_csv("results/final_submission_1st_place.csv", index=False)
print("Mass! Submission file saved: results/final_submission_1st_place.csv")