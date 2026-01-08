
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

print("--- BENCHMARK CONTEXTUALIZATION ---")

# 1. Load Data (Same clean/sort as before)
Y_raw = pd.read_csv("data/Training_Output.csv").set_index("ID").sort_index()["Target"]
split_point = int(len(Y_raw) * 0.8)
Y_train = Y_raw.iloc[:split_point]
Y_test = Y_raw.iloc[split_point:]

# 2. Define Metric
def napoleon_metric(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

print(f"Test Set Stats: Mean={Y_test.mean():.4f}, Std={Y_test.std():.4f}, Min={Y_test.min():.4f}, Max={Y_test.max():.4f}")
print("-" * 30)

# 3. Benchmark 1: The "Dumb" Static Mean (What gave 4.37)
# Problème : Le Train (1.26) est très loin du Test (~0.20)
bench_static = Y_train.mean()
pred_static = np.full_like(Y_test, bench_static)
score_static = napoleon_metric(Y_test, pred_static)
print(f"1. Static Benchmark (Train Mean): {score_static:.4f} (Predicts {bench_static:.4f} constant)")

# 4. Benchmark 2: The "Oracle" (If we knew the future mean)
# C'est la meilleure constante possible (La moyenne du Test)
bench_oracle = Y_test.mean()
pred_oracle = np.full_like(Y_test, bench_oracle)
score_oracle = napoleon_metric(Y_test, pred_oracle)
print(f"2. Oracle Benchmark (Test Mean):  {score_oracle:.4f} (Predicts {bench_oracle:.4f} constant)")

# 5. Benchmark 3: Naive Persistence (Last Value)
# On prédit que "Demain = Aujourd'hui"
# Shift 1: L'observation t prédite par t-1 (du Test set)
# Pour le premier point du test, on utilise le dernier du train
last_train = Y_train.iloc[-1]
pred_naive = np.concatenate(([last_train], Y_test.values[:-1]))
score_naive = napoleon_metric(Y_test, pred_naive)
print(f"3. Naive Persistence (Last Value): {score_naive:.4f}")

print("-" * 30)
print("Conclusion:")
print(f"Global Mean (4.37) is bad because Regime Shift.")
print(f"Target 'Low Phase' Benchmark is closer to {score_oracle:.4f}.")
print(f"Your Model (1.57) is competing against these.")
