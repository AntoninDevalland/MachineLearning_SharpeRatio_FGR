
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

print("Loading data for fix verification...")
try:
    X_raw = pd.read_csv("data/Training_Input.csv", index_col="ID")
    y_raw = pd.read_csv("data/Training_Output.csv", index_col="ID")
except FileNotFoundError:
    print("Error: Files not found.")
    exit(1)

# Sort by ID assuming ID corresponds to time (0 is earliest)
# IF input isn't sorted, we should sort it. The notebook ID seems sequential.
X_raw = X_raw.sort_index()
y_raw = y_raw.sort_index()

print("Using TimeSeriesSplit (Robust against leakage)...")
# TimeSeriesSplit creates expanding windows: Train on [0..k], Test on [k+1..k+n]
# This prevents future-to-past leakage.
tscv = TimeSeriesSplit(n_splits=5)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("sel", SelectFromModel(ElasticNet(alpha=0.1, l1_ratio=0.5))),
    ("model", ElasticNet(random_state=42))
])

scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw, y_raw)):
    X_tr, y_tr = X_raw.iloc[train_idx], y_raw.iloc[train_idx]
    X_val, y_val = X_raw.iloc[val_idx], y_raw.iloc[val_idx]
    
    pipe.fit(X_tr, y_tr.values.ravel())
    preds = pipe.predict(X_val)
    
    mae = mean_absolute_error(y_val, preds)
    scores.append(mae)
    print(f"Fold {fold+1} MAE: {mae:.4f}")

print(f"\nAverage TimeSeries CV MAE: {np.mean(scores):.4f}")
print("This score should be closer to your OOS performance (likely higher MAE than the leaked version).")
