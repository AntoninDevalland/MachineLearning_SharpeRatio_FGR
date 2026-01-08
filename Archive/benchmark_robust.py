
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error

print("Loading data...")
try:
    X_raw = pd.read_csv("data/Training_Input.csv", index_col="ID")
    y_raw = pd.read_csv("data/Training_Output.csv", index_col="ID")
except: 
    print("Error loading data")
    exit()

X_raw = X_raw.sort_index()
y_raw = y_raw.sort_index()
y = y_raw["Target"]

# --- Sample Weights (Regime Shift) ---
decay_rate = 0.0005
sample_weights = np.exp(-decay_rate * (len(y) - 1 - np.arange(len(y))))
sample_weights = sample_weights / sample_weights.mean()

# --- Models ---
models = {
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42)
}

# --- Validation ---
tscv = TimeSeriesSplit(n_splits=5)

results = {}

print(f"{'Model':<15} | {'Mean TimeSeries MAE':<20}")
print("-" * 40)

for name, model in models.items():
    scores = []
    
    # Simple Loop for Validation
    for train_idx, val_idx in tscv.split(X_raw, y):
        X_tr, y_tr = X_raw.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X_raw.iloc[val_idx], y.iloc[val_idx]
        w_tr = sample_weights[train_idx]
        
        # Fit with weights if supported
        try:
            if name in ["RandomForest", "ElasticNet"]:
                 model.fit(X_tr, y_tr, sample_weight=w_tr)
            elif name == "XGBoost":
                 model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            else:
                 model.fit(X_tr, y_tr)
        except:
             model.fit(X_tr, y_tr) # Fallback
            
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        scores.append(mae)
    
    avg_score = np.mean(scores)
    results[name] = avg_score
    print(f"{name:<15} | {avg_score:.4f}")

print("-" * 40)
best_model = min(results, key=results.get)
print(f"🏆 Best Robust Model: {best_model} ({results[best_model]:.4f})")
