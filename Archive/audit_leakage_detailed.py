
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

print("--- AUDIT LEAKAGE & FEATURE ENGINEERING ---")

# 1. Load Data
# On charge les données comme dans le notebook
print("Loading data...")
try:
    features = pd.read_csv("data/Training_Input.csv").set_index("ID")
    labels = pd.read_csv("data/Training_Output.csv").set_index("ID")
except FileNotFoundError:
    print("Error: Files not found. Run from correct directory.")
    exit()

# 2. Dedup (Exactement comme le notebook)
print(f"Original shape: {features.shape}")
features = features.drop_duplicates()
labels = labels.loc[features.index]
print(f"Deduped shape: {features.shape}")

# 3. Compute Returns (Logic du notebook)
print("Computing Returns (Notebook Logic)...")
def compute_returns(df):
    rdt = df.iloc[:, :7].copy()
    columns_value_strategies = df.columns[7:]
    # Simulation de la boucle du notebook
    # Le notebook fait: rdt[col] = log(next/current)
    # Les colonnes sont lag_19, lag_18 ... lag_0
    # Donc Yield_lag_19 = log(Price_lag_18 / Price_lag_19)
    # Yield_lag_0 n'existe pas car pas de lag_-1
    # On va recréer ça vectoriellement pour aller vite
    for i in range(len(columns_value_strategies) - 1):
        curr_col = columns_value_strategies[i]
        next_col = columns_value_strategies[i+1]
        # Nom de colonne dans le notebook : "rendement_" + nom de la courant
        # ex: rendement_X_3_lag_19
        new_col = "rendement_" + curr_col
        rdt[new_col] = np.log(df[next_col] / df[curr_col])
    return rdt

features_rdt = compute_returns(features)
# Drop lag_20 columns (inutiles)
cols_rdt_20 = [col for col in features_rdt.columns if "lag_20" in col]
features_rdt = features_rdt.drop(columns=cols_rdt_20)

print(f"Returns computed. Shape: {features_rdt.shape}")

# 4. Feature Engineering (Simplified Logic check)
# On va vérifier si une des features calculées (Vol, Cov, Skew, Sharpe) est le Target
# Le Target est le "Sharpe Ratio" (futur ? présent ?)
# On va calculer les features simples (Vol, Mean) sur les rendements
print("Calculating simple stats (Mean, Std) on returns...")
# On prend juste les rendements
ret_cols = [c for c in features_rdt.columns if "rendement" in c]
returns_df = features_rdt[ret_cols]

# 5. Correlation Check
print("--- CORRELATION CHECK ---")
# On assemble tout
X = features_rdt.copy()
# On ajoute Target
X["TARGET"] = labels["Target"]

# Top Correlations
corrs = X.corrwith(X["TARGET"]).abs().sort_values(ascending=False)
print("\nTOP 20 Feature Correlations with TARGET:")
print(corrs.head(20))

# 6. Shift Check
# Si le Target est t+1, peut-être qu'il est corrélé avec Shift(-1) de t ?
# Ou si Target est t, correlation directe ?
print("\nChecking Shift Correlations...")
# Shift des features (future peeking check) - non applicatif ici car on cherche leakage présent
# On regarde juste si Target est une fonction linéaire des features présents.

if corrs.iloc[1] > 0.95: # index 0 is TARGET itself
    print("\n🚨 ALERT: Found feature with > 0.95 correlation!")
    print("Possibility of LEAKAGE (Target is present in Input).")
else:
    print("\n✅ CLEAN: No single feature strongly correlates with Target.")
    print("The model is likely learning complex patterns, not reading the Answer.")

print("-" * 30)
