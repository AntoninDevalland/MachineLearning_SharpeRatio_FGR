
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import randint, uniform

# --- 1. Load Data ---
print("Loading data...")
try:
    X_raw = pd.read_csv("data/Training_Input.csv", index_col="ID")
    y_raw = pd.read_csv("data/Training_Output.csv", index_col="ID")
except FileNotFoundError:
    print("Error: Files not found.")
    exit(1)

X_raw = X_raw.sort_index()
y_raw = y_raw.sort_index()
y = y_raw["Target"]

# --- 2. Sample Weights (Regime Shift) ---
# Give more weight to recent data to adapt to the new market regime
decay_rate = 0.0005
sample_weights = np.exp(-decay_rate * (len(y) - 1 - np.arange(len(y))))
sample_weights = sample_weights / sample_weights.mean()

# --- 3. Pipeline ---
# Use Lasso for Feature Selection (removes noise)
# Use RandomForest for the final Prediction (handles non-linearity)
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("selector", SelectFromModel(Lasso(alpha=0.01, random_state=42), threshold="median")),
    ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))
])

# --- 4. Tuning Grid ---
# We optimize the Random Forest parameters using TimeSeries Validation
param_dist = {
    "selector__estimator__alpha": uniform(0.005, 0.05), # Selection strictness
    "regressor__n_estimators": randint(50, 150),
    "regressor__max_depth": randint(3, 10),  # Keep depth checks effectively
    "regressor__min_samples_leaf": randint(2, 10)
}

tscv = TimeSeriesSplit(n_splits=5)

print("Running TimeSeriesCV Tuning on RANDOM FOREST (Optimizing for Future)...")

# --- 4a. Baseline Check ---
dummy_scores = []
for tr, val in tscv.split(X_raw, y):
    dummy_scores.append(mean_absolute_error(y.iloc[val], np.full_like(y.iloc[val], y.iloc[tr].mean())))
print(f"[BASELINE] Mean-Predictor MAE: {np.mean(dummy_scores):.4f}\n")

# --- 5. Run Search ---
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=15, # Reasonable for demo, increase for production
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# RandomForest supports sample_weight in fit
# We pass it via fit_params. Note: The 'selector' (Lasso) usually doesn't need it or assumes scaled X.
# But providing it to the pipeline steps that support it is key.
fit_params = {
    "regressor__sample_weight": sample_weights,
    #"selector__sample_weight": sample_weights # Lasso step might struggle with pipelined weights in older sklearn
}

try:
    search.fit(X_raw, y, **fit_params)
except:
    print("Weight passing failed, fitting without weights...")
    search.fit(X_raw, y)

print(f"\n✅ Best Robust RF MAE: {-search.best_score_:.4f}")
print(f"Best Params: {search.best_params_}")

# --- 6. Final Model ---
best_model = search.best_estimator_
best_model.fit(X_raw, y, **fit_params) # Retrain on all data
n_feat = np.sum(best_model.named_steps["selector"].get_support())
print(f"Final model selected {n_feat} features.")
print("Feature Importances available in best_model.named_steps['regressor'].feature_importances_")
