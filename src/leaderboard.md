# 🏆 RT Activity Predictor - Leaderboard

This leaderboard tracks the performance of various models trained to predict Reverse Transcriptase (RT) activity.

| Rank | Model Description | Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **1** | **XGBoost + Optuna Tuning + Class Weighting** | **80.70%** | **Current Best** |
| 2 | Standard XGBoost (Handcrafted Features) | 78.10% | Baseline |
| 3 | Linear Model (Logistic Regression) | 74.00% | Deprecated |
| 4 | Hybrid Model (Handcrafted + ESM) | 71.50% | Experimental |

---

### Key Improvements in Top Model:
- **Optimization:** Hyperparameter tuning using Optuna for 100+ trials.
- **Handling Imbalance:** Used `scale_pos_weight` (2.03) to better detect active sequences.
- **Feature Selection:** Removed leakage columns (FoldSeek) to ensure real-world accuracy.
- **Model Path:** `models/final_xgb_optuna.pkl`