
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

print("Loading data...")
try:
    # Load just enough data to prove the point
    features = pd.read_csv("data/Training_Input.csv")
    features = features.sort_values("ID").set_index("ID")
except:
    print("Error loading data.")
    exit()

# --- 1. Recreate the Flawed Grouping Logic ---
# Taken directly from the notebook
print("Recreating Notebook Grouping Logic...")
features["I_1_lag_1_r"] = features["I_1_lag_1"].round(6)
grp = features.groupby("I_1_lag_1_r").groups

features["Group"] = -1
for ngroup, idx in enumerate(grp.values()):
    features.loc[idx, "Group"] = ngroup

groups = features["Group"].astype(int).values
obs_index = features.index.to_numpy()

# --- 2. Perform the Split ---
print("Running GroupShuffleSplit (as done in notebook)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(obs_index, groups=groups))

train_ids = obs_index[train_idx]
test_ids = obs_index[test_idx]

print(f"Train Size: {len(train_ids)}")
print(f"Test Size:  {len(test_ids)}")

# --- 3. Check for Overlap (The Proof) ---
# The Target is based on 21-day returns.
# So if Test ID X is time T, and Train has any data from T-20 to T+20, it's leakage.
# Since IDs are sequential integers representing time steps (mostly):

print("\nChecking for Temporal Leakage...")
leakage_count = 0
overlap_window = 20 # 21 days trajectory means +/- 20 days overlap

# Convert to set for O(1) lookup
train_id_set = set(train_ids)

for tid in test_ids:
    # Check if any "neighbor" exists in Train
    # We look for ID-20 to ID+20
    is_leaked = False
    for offset in range(-overlap_window, overlap_window + 1):
        if offset == 0: continue
        if (tid + offset) in train_id_set:
            is_leaked = True
            break
    
    if is_leaked:
        leakage_count += 1

leakage_pct = (leakage_count / len(test_ids)) * 100

print(f"\nRESULTS:")
print(f"Test Samples checked: {len(test_ids)}")
print(f"Leaked Samples:       {leakage_count}")
print(f"Leakage Percentage:   {leakage_pct:.2f}%")

if leakage_pct > 0:
    print("\nCONCLUSION: PROVEN.")
    print(f"For {leakage_pct:.1f}% of your Test set, the model has access to overlapping data in the Train set.")
    print("This confirms that GroupShuffleSplit behaves like random shuffling on this dataset.")
else:
    print("\nCONCLUSION: No leakage found (unexpected).")
