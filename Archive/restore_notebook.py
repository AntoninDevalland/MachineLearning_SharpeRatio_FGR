
import json
import os

nb_path = "predofshapr_nap.ipynb"
print(f"Loading {nb_path}...")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
patched_split = False
patched_cv = False

for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    
    source_str = "".join(cell["source"])
    
    # Target 1: The New Time Split Cell (Find based on my comments)
    if "# --- SÉPARATION TEMPORELLE STRICTE" in source_str:
        print(f"Found Patched Split Cell at index {i}. Reverting...")
        cell["source"] = [
            "features_clean = features_clean.copy()\n",
            "\n",
            "# Création des groupes (une trajectoire 21j = un groupe)\n",
            "features_clean[\"I_1_lag_1_r\"] = features_clean[\"I_1_lag_1\"].round(6)\n",
            "grp = features_clean.groupby(\"I_1_lag_1_r\").groups\n",
            "features_clean[\"Group\"] = -1\n",
            "for ngroup, idx in enumerate(grp.values()):\n",
            "    features_clean.loc[idx, \"Group\"] = ngroup\n",
            "\n",
            "groups = features_clean[\"Group\"].astype(int).values\n",
            "obs_index = features_clean.index.to_numpy()\n",
            "\n",
            "# Séparation du test set par groupes (anti data leakage)\n",
            "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n",
            "trainval_pos, test_pos = next(gss.split(obs_index, groups=groups))\n",
            "\n",
            "trainval_idx = obs_index[trainval_pos]\n",
            "test_idx = obs_index[test_pos]\n",
            "\n",
            "# Application du split aux features et aux labels\n",
            "X_train = features_total.loc[trainval_idx]\n",
            "X_test = features_total.loc[test_idx]\n",
            "\n",
            "Y_train = labels_clean.loc[trainval_idx]\n",
            "Y_test     = labels_clean.loc[test_idx]\n",
            "\n",
            "groups_trainval = features_clean.loc[trainval_idx, \"Group\"].values\n",
            "\n",
            "print(X_test.info())\n",
            "print(Y_test.info())\n"
        ]
        patched_split = True

    # Target 2: The TimeSeriesSplit Cell
    if "# Validation croisée Temporelle (TimeSeriesSplit)" in source_str:
        print(f"Found Patched CV Cell at index {i}. Reverting...")
        cell["source"] = [
             "# Validation croisée groupée\n",
             "cv = GroupKFold(n_splits=cfg.n_splits)\n"
        ]
        patched_cv = True

if patched_split and patched_cv:
    print("Writing restored notebook...")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Success! Notebook reverted.")
else:
    print(f"Failed to find targeted patched cells. Split:{patched_split}, CV:{patched_cv}")
