import pandas as pd
import xgboost as xgb
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

# 1. Load Data
print("Loading data for Optuna optimization...")
features_df = pd.read_csv('data/handcrafted_features.csv')
sequences_df = pd.read_csv('data/rt_sequences.csv')
df = features_df.merge(sequences_df[['rt_name', 'active']], on='rt_name', how='inner')

X = df.drop(['rt_name', 'active'], axis=1)
y = df['active']
foldseek_cols = [c for c in X.columns if "foldseek" in c]
X = X.drop(columns=foldseek_cols)

# 2. Define the Objective Function
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 3),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**param)
    
    # 5-Fold Cross Validation panni accuracy check pannuvom
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score

# 3. Run Optimization
print("Starting Optuna Study... This might take a minute.")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # 50 trials is a good start

print("\n--- Optimization Finished ---")
print(f"Best Accuracy Score: {study.best_value:.4f}")
print("Best Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# 4. Final Plot (Optional - Visualize optimization)
# optuna.visualization.plot_optimization_history(study).show()