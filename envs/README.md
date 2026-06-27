# Conda environment for labs

Единое окружение для всех лабораторных работ создается из файла `env-labs.yml`.
Окружение ставится в локальную папку проекта `.conda/envs/env-labs`, потому что
системный раздел с домашним каталогом почти заполнен, а на `/data` достаточно
места для тяжелых пакетов вроде PyTorch.

```bash
CONDA_PKGS_DIRS="$PWD/.conda/pkgs" conda env create -p "$PWD/.conda/envs/env-labs" -f envs/env-labs.yml
conda activate "$PWD/.conda/envs/env-labs"
python scripts/check_env.py
python -m ipykernel install --user --name env-labs --display-name "Python (env-labs)"
```

Окружение покрывает:

- интерпретацию моделей: `lime`, `shap`;
- методы для дисбаланса классов: `imbalanced-learn`, `xgboost`, `lightgbm`, `optuna`;
- графовые лабораторные: `torch`, `torch_geometric`;
- Node2Vec-рекомендации: `networkx`, `node2vec`, `gensim`;
- notebooks и визуализацию: `jupyterlab`, `matplotlib`, `seaborn`, `plotly`.

