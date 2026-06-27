# Methods for Processing and Analyzing Heterogeneous Data

A collection of laboratory assignments covering classical and modern machine learning techniques for heterogeneous data — tabular, graph-structured, and text modalities.

## Labs

| # | Folder | Topic |
|---|--------|-------|
| 1 | `Lab1/` | Preprocessing, modeling, and data drift detection on Titanic & Iris |
| 2 | `Imbalanced_data/` | Class imbalance handling: undersampling, oversampling, combined methods |
| 3 | `AI_interp/` | Model interpretation with LIME and SHAP on heterogeneous features |
| 4 | `GNN_GCN/` | Node classification: MLP vs GNN vs GCN on Facebook Page-Page dataset |
| 5 | `Node2Vec/` | Graph-based movie recommendation system using Node2Vec on MovieLens 100K |

## Structure

```
.
├── Lab1/                   # Preprocessing pipelines, outlier detection, scaling, drift
├── Imbalanced_data/        # Imbalanced classification with imblearn / XGBoost
├── AI_interp/              # LIME & SHAP interpretation of multimodal models
├── GNN_GCN/                # PyTorch Geometric: MLP / GNN / GCN comparison
├── Node2Vec/               # Node2Vec embeddings + film recommender
├── docs/                   # Lecture notes (Markdown)
├── envs/                   # Conda environment definition
└── scripts/                # Utility scripts (env check, etc.)
```

## Environment Setup

A single conda environment covers all labs:

```bash
CONDA_PKGS_DIRS="$PWD/.conda/pkgs" conda env create -p "$PWD/.conda/envs/env-labs" -f envs/env-labs.yml
conda activate "$PWD/.conda/envs/env-labs"
python scripts/check_env.py
python -m ipykernel install --user --name env-labs --display-name "Python (env-labs)"
```

Key packages: `torch`, `torch_geometric`, `networkx`, `node2vec`, `shap`, `lime`, `imbalanced-learn`, `xgboost`, `lightgbm`, `jupyterlab`.

## Lab Summaries

### Lab 1 — Heterogeneous Data Preprocessing
Tabular preprocessing pipeline benchmark on Titanic and Iris datasets.  
Compares imputation strategies (`mean`, `median`, `knn`, `iterative`, …), outlier handling (`winsorization`, `log`, `boxcox`, …), and feature scaling (`standard`, `minmax`, `robust`, …) via cross-validation. Includes KS-test based data drift report.

**Run:** `python Lab1/lab1.py [--dataset 1|2] [--max-rows N] [--use-gpu]`

### Lab 2 — Imbalanced Data
Studies the effect of class imbalance on model performance and compares balancing strategies: random undersampling, SMOTE oversampling, and combined approaches. Evaluated with `accuracy` and `balanced accuracy`.

**Notebook:** `Imbalanced_data/imbalanced_data_methods.ipynb`

### Lab 3 — AI Model Interpretation
Trains a binary classifier (Random Forest / Gradient Boosting) on synthetic heterogeneous data — tabular features + text embeddings. Applies LIME for local explanation and SHAP for local/global feature importance. Aggregates contributions per modality.

**Notebook:** `AI_interp/interpretation_ai_models.ipynb`

### Lab 4 — GNN / GCN on Graph Data
Implements and compares three architectures for node classification on the **Facebook Page-Page** graph:
- **MLP** — node features only, no graph structure
- **GNN** — message passing from neighbors
- **GCN** — graph convolutional aggregation

**Notebook:** `GNN_GCN/mlp_gnn_gcn_facebook.ipynb`

> **Dataset:** downloaded automatically via `torch_geometric`. Raw files are excluded from the repo (see `.gitignore`).

### Lab 5 — Node2Vec Film Recommender
Builds a weighted film co-preference graph from MovieLens 100K (edges = ≥20 users liked both films), trains Node2Vec embeddings, and uses nearest neighbors in embedding space for recommendations.  
Expected graph: 410 nodes, 14 936 edges.

**Notebook:** `Node2Vec/film_recommendation_node2vec.ipynb`

> **Dataset:** download from https://files.grouplens.org/datasets/movielens/ml-100k.zip and place in `Node2Vec/ml-100k/`.
