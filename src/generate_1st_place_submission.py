import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import os

# 1. Folders Check
for folder in ['models', 'results']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 2. Data Load and Clean
# Note: Path is 'data/...' because we run from root
features_df = pd.read_csv("data/handcrafted_features.csv")
sequences_df = pd.read_csv("data/rt_sequences.csv")
df = features_df.merge(sequences_df[["rt_name", "active"]], on="rt_name", how="inner")

X = df.drop(["rt_name", "active"], axis=1)
y = df["active"]
X = X.drop(columns=[c for c in X.columns if "foldseek" in c], errors='ignore')

# 3. Robust NaN/Inf Handling
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

# 4. Train and Calibrate (Hema's Secret for 1st Place)
base_model = XGBClassifier(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=6, 
    use_label_encoder=False, 
    eval_metric='logloss'
)

print("Training Calibrated Model for Hema...")
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated_model.fit(X, y)

# 5. Save the Winning Model
joblib.dump(calibrated_model, "models/final_xgb_optuna_calibrated.pkl")

# 6. Generate Submission with Probability Clipping
probs = calibrated_model.predict_proba(X)[:, 1]
probs = np.clip(probs, 0.01, 0.99) # This boosts CLS score!

submission = pd.DataFrame({
    'rt_name': features_df['rt_name'],
    'probability': probs
})

submission.to_csv("results/final_submission_1st_place.csv", index=False)
print("MASS! 'results/final_submission_1st_place.csv' is ready!")