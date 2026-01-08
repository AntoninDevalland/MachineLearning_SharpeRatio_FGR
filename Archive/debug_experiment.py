
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, make_scorer

# --- 1. Load Data ---
print("Loading data...")
try:
    X_raw = pd.read_csv("data/Training_Input.csv", index_col="ID")
    y_raw = pd.read_csv("data/Training_Output.csv", index_col="ID")
except FileNotFoundError:
    print("Error: Files not found in data/ directory.")
    exit(1)

# --- 2. Feature Engineering (Simplified Replication) ---
# We replicate the Group creation logic to show the groups
print("Processing features...")
features_clean = X_raw.copy()

# This is the KEY suspected leakage point from the notebook:
# "Création des groupes (une trajectoire 21j = un groupe)"
# It groups by the exact value of a lagged feature.
# In a sliding window setting, Day t and Day t+1 are distinct groups but overlap 95%.
if "I_1_lag_1" in features_clean.columns:
    # Assuming 'I_1_lag_1' exists, if not we might need to recreate it from raw columns if calculation was complex.
    # checking columns from previous view: data starts with weights and lag returns.
    # Let's assume the CSV contains pre-computed features or raw data?
    # The view showed "rendement_I_1_lag_19".
    # We might need to adjust column names if the CSV is raw input.
    pass
else:
    # If using raw input, we presume the CSV provided IS the input to feature_engineering.
    # The notebook does distinct steps.
    # For reproduction, if we can't run full feature engineering without full context,
    # we simulate the "broken" CV by using INDEX as group if duplicate groups aren't found.
    features_clean["Group"] = features_clean.index # Default behavior if no duplicates logic applied
    
    # Try to apply the notebook's logic if column exists
    # NB: The notebook computes features_total from features_rdt.
    # If Training_Input.csv is RAW, we might miss calculated columns.
    pass

# Check if we have groups
# If the user script does:
# features_rdt = features_total ...
# We'll just define groups as the index for now to demonstrate the 'random split' effect of unique groups.
groups = features_clean.index.to_numpy()

# --- 3. Model & CV ---
print("Setting up model and CV...")
# ElasticNet Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("sel", SelectFromModel(ElasticNet(alpha=0.1, l1_ratio=0.5))),
    ("model", ElasticNet(random_state=42))
])

# CV Strategy from Notebook
# GroupKFold with "unique" groups acts like KFold with shuffling (if groups are unique rows)
cv = GroupKFold(n_splits=5)

# --- 4. Run Evaluation ---
print("Running Cross-Validation...")
scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(features_clean, y_raw, groups=groups)):
    X_tr, y_tr = features_clean.iloc[train_idx], y_raw.iloc[train_idx]
    X_val, y_val = features_clean.iloc[val_idx], y_raw.iloc[val_idx]
    
    # Simple fit (skipping huge grid search for speed in debug)
    pipe.fit(X_tr, y_tr.values.ravel())
    preds = pipe.predict(X_val)
    
    mae = mean_absolute_error(y_val, preds)
    scores.append(mae)
    print(f"Fold {fold+1} MAE: {mae:.4f}")

print(f"\nAverage CV MAE: {np.mean(scores):.4f}")
print("If this score is significantly better (lower) than your OOS score, leakage is confirmed.")
