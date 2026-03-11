import pandas as pd
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
import numpy as np

print("Loading data files...")
# 1. Loading all necessary files
features_df = pd.read_csv('data/handcrafted_features.csv')
sequences_df = pd.read_csv('data/rt_sequences.csv')

# 2. Merging features with target (active) and family
# We use 'rt_name' as the key to join files
df = features_df.merge(sequences_df[['rt_name', 'active', 'rt_family']], on='rt_name', how='inner')

print(f"Total samples after merging: {len(df)}")

# 3. Setup Features (X), Target (y), and Groups
# 'rt_name', 'active', and 'rt_family' are not features for the model
X = df.drop(['rt_name', 'active', 'rt_family'], axis=1)
y = df['active']
groups = df['rt_family']

# 4. XGBoost Model Configuration
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 5. Leave-One-Family-Out (LOFO) Validation
logo = LeaveOneGroupOut()
scores = []

print(f"Training on {X.shape[1]} features using LOFO...")

for train_idx, test_idx in logo.split(X, y, groups):
    # Split data based on family
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train and Predict
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Calculate F1 Score for this family
    score = f1_score(y_test, preds, average='macro')
    scores.append(score)
    
    family_name = groups.iloc[test_idx].unique()[0]
    print(f"Family: {family_name:25} | F1 Score: {score:.4f}")

print("\n" + "="*35)
print(f"Final Average XGBoost F1 Score: {np.mean(scores):.4f}")
print("Target to beat: 0.5330")
print("="*35)