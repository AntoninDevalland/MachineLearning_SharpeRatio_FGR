
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, make_scorer
from types import SimpleNamespace

# --- PRE-REQUISITES (recreating notebook context) ---
print("--- 1. Environment Setup ---")
cfg = SimpleNamespace(seed=42, n_iter_fast=10, n_splits=5)

def napoleon_metric(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

napoleon_scorer = make_scorer(napoleon_metric, greater_is_better=False)

def evaluate_cv_sklearn(estimator, X_feat, y, groups, cv, metric_fn, sample_weight=None):
    # Simplified version of the notebook function for verification
    fold_scores = []
    oof = np.zeros(len(y))
    for tr, va in cv.split(X_feat, y, groups=groups):
        X_tr, y_tr = X_feat.iloc[tr], y.iloc[tr]
        X_va, y_va = X_feat.iloc[va], y.iloc[va]
        
        # Fit with weights if provided
        if sample_weight is not None:
             estimator.fit(X_tr, y_tr, model__sample_weight=sample_weight[tr])
        else:
             estimator.fit(X_tr, y_tr)
             
        pred = estimator.predict(X_va).reshape(-1)
        oof[va] = pred
        s = metric_fn(y_va.values, pred)
        fold_scores.append(s)
    
    return {"mean": np.mean(fold_scores), "std": np.std(fold_scores)}

def store_eval(model, variant, eval_out):
    pass # Dummy

# --- DATA LOADING (CLEAN) ---
print("--- 2. Data Loading ---")
X_raw = pd.read_csv("data/Training_Input.csv").set_index("ID").sort_index()
Y_raw = pd.read_csv("data/Training_Output.csv").set_index("ID").sort_index()

# Extract Target
Y_raw = Y_raw["Target"] # Series

# Split 80%
split_point = int(len(X_raw) * 0.8)
X_train = X_raw.iloc[:split_point]
X_test = X_raw.iloc[split_point:]
Y_train = Y_raw.iloc[:split_point] # Series !! The user snippet implies Y_train might be DataFrame?
# Wait, user snippet has: "pred_test_fs = rf_fs_best.predict(X_test)" and "napoleon_metric(Y_test.values, pred_test_fs)"
# If Y_train is DataFrame, RandomizedSearchCV needs y in fit.
# Let's keep Y_train as DataFrame because user checks Y_train["Target"] inside notebook usually?
# Actually in the snippet: "rf_fs_search.fit(X_train, Y_train...)"
# Sklearn usually expects Series. But let's check notebook behavior.
# Notebook line 2234: "y = Y_train['Target']" implying Y_train IS a DataFrame.
# BUT sklearn fit(X, y) usually warns if y is DataFrame.
# Train_robust.py uses Series.
# Let's try keeping it as Series for safety, usually safer.
# Wait, user snippet uses "Y_test.values". If Y_test is Series, .values works.
# Let's force Series.
Y_train = Y_train # Is Series
Y_test = Y_test   # Is Series

# Groups
groups_trainval = np.arange(len(X_train))

# Sample Weights
decay_rate = 0.0005
sample_weights = np.exp(-decay_rate * (len(Y_train) - 1 - np.arange(len(Y_train))))
sample_weights = sample_weights / sample_weights.mean()

# CV
cv = TimeSeriesSplit(n_splits=cfg.n_splits)

# --- USER SNIPPET EXECUTION ---
print("--- 3. Running User Snippet ---")

rf_fs = Pipeline([
    ("prep", RobustScaler()),  # RobustScaler est essentiel ici
    # On utilise LASSO pour choisir les variables (comme dans train_robust.py)
    ("sel", SelectFromModel(Lasso(random_state=cfg.seed), threshold="median")),
    ("model", RandomForestRegressor(random_state=cfg.seed, n_jobs=-1))
])
rf_fs_space = {
    # Alpha contrôle la sévérité du tri (plus grand = plus sévère)
    "sel__estimator__alpha": [0.005, 0.01, 0.02, 0.05],
    "model__n_estimators": [100, 300, 600],
    "model__max_depth": [6, 10, 15],
    "model__min_samples_leaf": [2, 5, 10]
}
# Lasso supporte mal les poids dans ce pipeline sklearn spécifique,
# donc on cible uniquement le modèle final pour les poids.
fit_params = {"model__sample_weight": sample_weights}
rf_fs_search = RandomizedSearchCV(
    rf_fs,
    param_distributions=rf_fs_space,
    n_iter=cfg.n_iter_fast,
    scoring=napoleon_scorer,
    cv=cv,
    random_state=cfg.seed,
    n_jobs=-1,
    verbose=1,
    refit=True
)
print("Fitting...")
rf_fs_search.fit(X_train, Y_train, groups=groups_trainval, **fit_params)

rf_fs_best = rf_fs_search.best_estimator_
print("Best params:", rf_fs_search.best_params_)

# Eval CV
# Note: evaluate_cv_sklearn implementation above is simplified, output might slightly differ in print format
rf_fs_eval = evaluate_cv_sklearn(
    rf_fs_best, X_train, Y_train, groups_trainval, cv, napoleon_metric, sample_weight=sample_weights
)
print("CV Mean (Train):", rf_fs_eval["mean"])

# TEST FINAL (OOS) - C'est ici qu'on attend 2.69
pred_test_fs = rf_fs_best.predict(X_test)
score = napoleon_metric(Y_test.values, pred_test_fs)
print("✅ SCORE OOS (RF + Lasso) :", score)
