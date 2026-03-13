## RT Activity Prediction Leaderboard

**Goal**: Predict whether a reverse transcriptase (RT) is active using sequence-derived handcrafted features, while avoiding FoldSeek-style leakage features.

### Overall Leaderboard (Handcrafted Features Only)

| Rank | Model                               | Features                    | Validation            | Accuracy | Notes                                   |
|------|-------------------------------------|-----------------------------|-----------------------|----------|-----------------------------------------|
| 1    | XGBoost (Optuna-tuned, this work)   | Handcrafted (no FoldSeek)   | Stratified 5-fold CV  | **0.8076** (80.76%) | Final champion model (`final_prediction_pipeline.py`) |
| 2    | RandomForest baseline               | Handcrafted (no FoldSeek)   | Stratified CV / LOFO  | ~0.78 (78%)         | Baseline from earlier experiments & plots |

### Provenance

- **Optuna search**: `src/optuna_tuning.py`  
  - Optimized an `xgboost.XGBClassifier` on handcrafted features with FoldSeek columns removed.  
  - The best trial achieved ~80.7–80.8% CV accuracy.

- **Final model & pipeline**: `src/final_prediction_pipeline.py`  
  - Reuses the best Optuna parameters and the same data-preprocessing logic.  
  - Provides:
    - `load_handcrafted_rt_data()` for consistent data loading and leakage removal.  
    - `get_best_xgb_model()` for instantiating the Optuna-tuned XGBoost.  
    - `cross_validated_report()` for Stratified K-fold metrics and confusion matrix.  
    - `train_final_model()` to fit on all data and save the model to `models/final_xgb_optuna.pkl`.  

- **Baselines**: `src/comparison_final.py`, `src/model_comparison.py`  
  - Compare RandomForest-based variants (handcrafted / ESM / hybrid) and earlier XGBoost baselines.  
  - The best RandomForest on handcrafted features reached approximately 78% accuracy, which the Optuna-tuned XGBoost now surpasses.

### How to Reproduce the Winning Model

1. Ensure data files are present:
   - `data/handcrafted_features.csv`
   - `data/rt_sequences.csv`

2. Run the final pipeline script:

   ```bash
   python -m src.final_prediction_pipeline
   ```

   This will:
   - Load and preprocess the handcrafted feature dataset (dropping FoldSeek columns).  
   - Report Stratified 5-fold cross-validated accuracy (≈0.807–0.808).  
   - Train the final XGBoost model on the full dataset and save it to `models/final_xgb_optuna.pkl`.

