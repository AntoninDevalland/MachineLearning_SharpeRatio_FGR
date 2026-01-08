
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
    
    # Target 1: The OOS Split Cell
    if "GroupShuffleSplit(n_splits=1, test_size=0.2" in source_str:
        print(f"Found Split Cell at index {i}. Patching...")
        cell["source"] = [
            "# --- SÉPARATION TEMPORELLE STRICTE (Fix Data Leakage) ---\n",
            "# On s'assure que les données sont triées par l'index (temps)\n",
            "features_clean = features_clean.sort_index()\n",
            "\n",
            "# Point de coupure à 80% (Train) / 20% (Test)\n",
            "split_point = int(len(features_clean) * 0.80)\n",
            "\n",
            "trainval_idx = features_clean.index[:split_point]\n",
            "test_idx = features_clean.index[split_point:]\n",
            "\n",
            "# Création d'un vecteur 'Group' factice pour compatibilité\n",
            "# Mais nous utiliserons TimeSeriesSplit qui ignore les groupes.\n",
            "features_clean['Group'] = features_clean.index\n",
            "\n",
            "# Application du split aux features et aux labels\n",
            "X_train = features_total.loc[trainval_idx]\n",
            "X_test = features_total.loc[test_idx]\n",
            "\n",
            "Y_train = labels_clean.loc[trainval_idx]\n",
            "Y_test     = labels_clean.loc[test_idx]\n",
            "\n",
            "groups_trainval = features_clean.loc[trainval_idx, 'Group'].values\n",
            "\n",
            "print(X_test.info())\n",
            "print(Y_test.info())\n"
        ]
        patched_split = True

    # Target 2: The Cross-Validation Definition Cell
    if "cv = GroupKFold(n_splits=cfg.n_splits)" in source_str:
        print(f"Found CV Cell at index {i}. Patching...")
        cell["source"] = [
            "# Validation croisée Temporelle (TimeSeriesSplit)\n",
            "# Assure que le fold de validation est toujours dans le futur du fold d'entrainement\n",
            "from sklearn.model_selection import TimeSeriesSplit\n",
            "cv = TimeSeriesSplit(n_splits=cfg.n_splits)\n"
        ]
        patched_cv = True

if patched_split and patched_cv:
    print("Writing patched notebook...")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Success!")
else:
    print(f"Failed to find all targets. Split:{patched_split}, CV:{patched_cv}")
