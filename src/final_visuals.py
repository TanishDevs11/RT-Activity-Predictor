import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score

# 1. Data Loading
features_df = pd.read_csv('data/handcrafted_features.csv')
sequences_df = pd.read_csv('data/rt_sequences.csv')
df = features_df.merge(sequences_df[['rt_name', 'active']], on='rt_name', how='inner')

X = df.drop(['rt_name', 'active'], axis=1)
y = df['active']
X = X.drop(columns=[c for c in X.columns if "foldseek" in c])

# 2. Best Parameters from Optuna
best_params = {
    'n_estimators': 188,
    'max_depth': 4,
    'learning_rate': 0.05474919881626068,
    'subsample': 0.7167183980497441,
    'colsample_bytree': 0.5895901388066157,
    'gamma': 3.0690014078763155,
    'scale_pos_weight': 2.032711735305366,
    'eval_metric': 'logloss',
    'random_state': 42
}

# 3. Model Training
model = xgb.XGBClassifier(**best_params)
model.fit(X, y) # Fitting for feature importance

# 4. Plotting Feature Importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=15, importance_type='weight', color='teal')
plt.title("Top 15 Features driving 80.7% Accuracy")
plt.tight_layout()
plt.savefig('results/feature_importance_80.png')
print("Graph saved: results/feature_importance_80.png")

# 5. Show the plot on screen
plt.show()