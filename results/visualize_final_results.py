import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

if not os.path.exists('results'):
    os.makedirs('results')

features_df = pd.read_csv("data/handcrafted_features.csv")
sequences_df = pd.read_csv("data/rt_sequences.csv")
df = features_df.merge(sequences_df[["rt_name", "active"]], on="rt_name", how="inner") 

X = df.drop(["rt_name", "active"], axis=1)
y = df["active"]
foldseek_cols = [c for c in X.columns if "foldseek" in c]
if foldseek_cols: X = X.drop(columns=foldseek_cols)

model = joblib.load("models/final_xgb_optuna.pkl")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=cv)

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Inactive', 'Active'], 
            yticklabels=['Inactive', 'Active'])
plt.title('Final Model: 80.7% Accuracy - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.savefig("results/model_performance_chart.png")
print("Success! Visual chart saved in results/model_performance_chart.png")
plt.show()