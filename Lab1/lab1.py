#!/usr/bin/env python3
"""Лабораторная работа 1: предобработка, нормализация, моделирование и мониторинг дрейфа на Titanic."""

from itertools import product
from pathlib import Path
import warnings

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, boxcox, yeojohnson
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    Normalizer,
)
from tqdm import tqdm
try:
    from cuml.ensemble import RandomForestRegressor as CuRandomForestRegressor
    from cuml.linear_model import LogisticRegression as CuLogisticRegression
except ImportError:  # GPU libs not available
    CuRandomForestRegressor = None
    CuLogisticRegression = None

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="n_quantiles.*is greater than.*samples")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._data")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Setting an item of incompatible dtype.*")

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MISSING_STRATEGIES = [
    "mean",
    "median",
    "knn",
    "most_frequent",
    "constant_zero",
    "iterative",
    "drop",
]
OUTLIER_METHODS = ["none", "winsorization", "log", "boxcox", "yeojohnson"]
SCALING_METHODS = ["none", "standard", "minmax", "robust", "quantile", "unit_vector"]

DATASETS = {
    "1": {
        "name": "Titanic",
        "description": "Passengers and survival outcome",
        "path": DATA_DIR / "Titanic-Dataset.csv",
        "classification_target": "survived",
        "regression_target": "fare",
        "classification_features": ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"],
        "classification_numeric": ["pclass", "age", "sibsp", "parch", "fare"],
        "classification_categorical": ["sex", "embarked"],
        "classification_allowed_values": [0, 1],
        "regression_features": ["pclass", "sex", "age", "sibsp", "parch", "embarked"],
        "regression_numeric": ["pclass", "age", "sibsp", "parch"],
        "regression_categorical": ["sex", "embarked"],
    },
    "2": {
        "name": "Iris",
        "description": "Iris flower dataset",
        "path": DATA_DIR / "Iris.csv",
        "classification_target": "species",
        "regression_target": "petal_length",
        "classification_features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classification_numeric": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classification_categorical": [],
        "classification_allowed_values": ["setosa", "versicolor", "virginica"],
        "regression_features": ["sepal_length", "sepal_width", "petal_width"],
        "regression_numeric": ["sepal_length", "sepal_width", "petal_width"],
        "regression_categorical": [],
    },
}

DEFAULT_DATASET_ID = "1"

CLASSIFICATION_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, average="weighted", zero_division=0),
    "recall": make_scorer(recall_score, average="weighted", zero_division=0),
    "f1": make_scorer(f1_score, average="weighted", zero_division=0),
}
REGRESSION_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def build_classification_estimator(use_gpu: bool):
    """Return CPU or GPU logistic regression."""
    if use_gpu and CuLogisticRegression is not None:
        return CuLogisticRegression()
    return LogisticRegression(solver="liblinear", max_iter=500, random_state=42)


def build_regression_estimator(use_gpu: bool):
    """Return CPU or GPU random forest regressor."""
    if use_gpu and CuRandomForestRegressor is not None:
        return CuRandomForestRegressor(n_estimators=100, random_state=42)
    return RandomForestRegressor(n_estimators=100, random_state=42)


def get_dataset_config(dataset_id: str):
    """Return metadata for the chosen dataset."""
    config = DATASETS.get(str(dataset_id))
    if config is None:
        raise ValueError(f"Unknown dataset `{dataset_id}`; choose from {list(DATASETS.keys())}")
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Lab1 preprocessing and evaluation runner.")
    parser.add_argument(
        "-d",
        "--dataset",
        choices=list(DATASETS.keys()),
        default=DEFAULT_DATASET_ID,
        help="Dataset selector by ID",
    )
    parser.add_argument(
        "-l",
        "--max-rows",
        type=int,
        default=None,
        help="Limit the number of rows read from the dataset (useful for large files)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Attempt to use cuML estimators on GPU (falls back to CPU if unavailable)",
    )
    return parser.parse_args()


def log_stage(name: str, status: str, info: str = ""):
    """Simple status indicator for the CLI."""
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    payload = f"{timestamp} | {name:<25} | {status}"
    if info:
        payload += f" | {info}"
    print(payload)


def load_dataset(config: dict, max_rows: int = None) -> pd.DataFrame:
    """Load the requested CSV and ensure required columns exist."""
    path = Path(config["path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset {config['name']} not found at {path}")
    raw = pd.read_csv(path)

    # Normalize column names from CamelCase to snake_case for Iris dataset
    if config["name"] == "Iris":
        raw.columns = raw.columns.str.replace('([a-z0-9])([A-Z])', r'\1_\2', regex=True)
        raw.columns = raw.columns.str.lower()
        raw.columns = raw.columns.str.replace('_cm', '', regex=True)  # Remove '_cm' suffix
        # Remove 'Iris-' prefix from species values
        if 'species' in raw.columns:
            raw['species'] = raw['species'].str.replace('Iris-', '', regex=True)
    else:
        raw.columns = raw.columns.str.lower()
    required = set(
        config["classification_features"]
        + config["regression_features"]
        + [config["classification_target"], config["regression_target"]]
    )
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df = raw.copy()
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
    # Normalize numeric columns
    numeric_cols = set(
        config["classification_numeric"] + config["regression_numeric"]
    )
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_drop_strategy(df, numeric_features, target_cols, strategy):
    """Drop rows when the 'drop' imputer strategy is selected."""
    if strategy != "drop":
        return df
    return df.dropna(subset=list(numeric_features) + list(target_cols))


def get_numeric_imputer(strategy: str):
    """Возвращает импутер для числовых признаков в соответствии со стратегией."""
    if strategy == "mean":
        return SimpleImputer(strategy="mean")
    if strategy == "median":
        return SimpleImputer(strategy="median")
    if strategy == "knn":
        return KNNImputer(n_neighbors=5)
    if strategy == "most_frequent":
        return SimpleImputer(strategy="most_frequent")
    if strategy == "constant_zero":
        return SimpleImputer(strategy="constant", fill_value=0)
    if strategy == "iterative":
        return IterativeImputer(random_state=42)
    if strategy == "drop":
        return SimpleImputer(strategy="mean")
    raise ValueError(f"Unsupported numeric imputer: {strategy}")


def get_scaler(name: str):
    """Возвращает скейлер или None, если масштабирование не нужно."""
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    if name == "unit_vector":
        return Normalizer(norm='l2')  # L2 нормировка для unit vector scaling
    if name == "none":
        return None
    raise ValueError(f"Unknown scaling option: {name}")


def detect_outliers(df: pd.DataFrame, numeric_cols, method="none") -> pd.DataFrame:
    """Применяет выбранную стратегию трансформации для обработки выбросов."""
    if method == "none":
        return df.copy()

    result_df = df.copy()
    for col in numeric_cols:
        series = df[col]
        if series.empty:
            continue

        # Обрабатываем только положительные значения для log, boxcox
        if method in ["log", "boxcox"]:
            min_val = series.min()
            if min_val <= 0:
                # Добавляем смещение для положительности
                shift = abs(min_val) + 1e-6
                series = series + shift

        if method == "winsorization":
            # Winsorization: заменяем экстремальные значения на квантили
            lower = series.quantile(0.05)
            upper = series.quantile(0.95)
            result_df[col] = series.clip(lower=lower, upper=upper).astype(series.dtype)
        elif method == "log":
            # Логарифмическое преобразование
            result_df[col] = np.log(series).astype(float)
        elif method == "boxcox":
            # Box-Cox преобразование
            try:
                transformed, _ = boxcox(series.dropna())
                result_df[col] = result_df[col].astype(float)
                result_df.loc[series.notna(), col] = transformed.astype(float)
            except ValueError:
                # Если Box-Cox не сработал, используем логарифм
                result_df[col] = result_df[col].astype(float)
                result_df[col] = np.log(series + abs(series.min()) + 1e-6).astype(float)
        elif method == "yeojohnson":
            # Yeo-Johnson преобразование
            try:
                transformed, _ = yeojohnson(series.dropna())
                result_df[col] = result_df[col].astype(float)
                result_df.loc[series.notna(), col] = transformed.astype(float)
            except ValueError:
                # Если Yeo-Johnson не сработал, оставляем без изменений
                pass
        else:
            raise ValueError(f"Unsupported outlier method: {method}")
    return result_df


def build_preprocessing_pipeline(
    numeric_features, categorical_features, numeric_imputer, scaler, estimator
) -> Pipeline:
    """Формирует колонн-трансформер и модель в едином пайплайне."""
    numeric_steps = [("imputer", numeric_imputer)]
    if scaler is not None:
        numeric_steps.append(("scaler", scaler))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])


def evaluate_classification(df, config, outlier_method, imputer_name, scaler_name, use_gpu=False):
    """Кросс-валидация логистической регрессии с разными вариантами препроцессинга."""
    target = config["classification_target"]
    features = config["classification_features"]
    numeric = config["classification_numeric"]
    categorical = config["classification_categorical"]
    prepared = apply_drop_strategy(df, numeric + categorical, [target], imputer_name)
    filtered = detect_outliers(prepared, numeric, method=outlier_method)
    filtered = filtered.dropna(subset=[target])
    allowed = config.get("classification_allowed_values")
    if allowed:
        filtered = filtered[filtered[target].isin(allowed)]
    if filtered.empty:
        raise ValueError("No data after outlier filtering.")

    X = filtered[features]
    y = filtered[target]
    numeric_imputer = get_numeric_imputer(imputer_name)
    scaler = get_scaler(scaler_name)
    pipeline = build_preprocessing_pipeline(
        numeric,
        categorical,
        numeric_imputer,
        scaler,
        build_classification_estimator(use_gpu),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=CLASSIFICATION_SCORING,
        n_jobs=1,
    )

    return {
        "Outlier": outlier_method,
        "Imputer": imputer_name,
        "Scaler": scaler_name,
        "Accuracy": results["test_accuracy"].mean(),
        "Precision": results["test_precision"].mean(),
        "Recall": results["test_recall"].mean(),
        "F1": results["test_f1"].mean(),
        "SampleCount": X.shape[0],
        "FoldCount": cv.get_n_splits(X, y),
    }


def evaluate_regression(df, config, outlier_method, imputer_name, scaler_name, use_gpu=False):
    """Кросс-валидация случайного леса с различными предобработками."""
    target = config["regression_target"]
    features = config["regression_features"]
    numeric = config["regression_numeric"]
    categorical = config["regression_categorical"]
    prepared = apply_drop_strategy(df, numeric + categorical, [target], imputer_name)
    filtered = detect_outliers(prepared, numeric, method=outlier_method)
    filtered = filtered.dropna(subset=[target])
    if filtered.empty:
        raise ValueError("No data after outlier filtering for regression.")

    X = filtered[features]
    y = filtered[target]
    numeric_imputer = get_numeric_imputer(imputer_name)
    scaler = get_scaler(scaler_name)
    estimator = build_regression_estimator(use_gpu)
    pipeline = build_preprocessing_pipeline(
        numeric,
        categorical,
        numeric_imputer,
        scaler,
        estimator,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=REGRESSION_SCORING, n_jobs=1
    )
    return {
        "Outlier": outlier_method,
        "Imputer": imputer_name,
        "Scaler": scaler_name,
        "RMSE": -results["test_rmse"].mean(),
        "MSE": -results["test_mse"].mean(),
        "MAE": -results["test_mae"].mean(),
        "R2": results["test_r2"].mean(),
        "SampleCount": X.shape[0],
        "FoldCount": cv.get_n_splits(X, y),
    }


def detect_drift(reference, current, numeric_features, threshold=0.05) -> pd.DataFrame:
    """Проводит KS-тест по каждому числовому признаку для поиска дрейфа."""
    records = []
    for col in numeric_features:
        ref_vals = reference[col].dropna()
        curr_vals = current[col].dropna()
        if ref_vals.empty or curr_vals.empty:
            records.append(
                {"feature": col, "p_value": np.nan, "drift": False, "note": "not enough samples"}
            )
            continue
        _, p_value = ks_2samp(ref_vals, curr_vals)
        records.append({"feature": col, "p_value": p_value, "drift": p_value < threshold, "note": ""})
    return pd.DataFrame(records).sort_values("p_value")


def ensure_figures_dir():
    """Обеспечивает существование папки для сохранения графиков."""
    FIGURES_DIR.mkdir(exist_ok=True)


def ensure_results_dir():
    """Обеспечивает существование папки для сохранения полных результатов."""
    RESULTS_DIR.mkdir(exist_ok=True)


def plot_missingness(df: pd.DataFrame, config: dict):
    """Сохраняет heatmap пропусков по колонкам."""
    subset = df[config["classification_features"] + [config["classification_target"]]]
    missing_matrix = subset.isnull().astype(int)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(missing_matrix, cmap='YlOrRd', cbar_kws={'label': 'Missing (1=Yes, 0=No)'},
                ax=ax, yticklabels=False)
    ax.set_title("Missing Values Heatmap")
    ax.set_xlabel("Features")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missing_values.png", bbox_inches="tight")
    plt.close(fig)


def plot_outlier_distributions(df: pd.DataFrame, config: dict):
    """Сохраняет boxplot для визуализации выбросов в числовых признаках."""
    numeric_cols = config["classification_numeric"][:4]  # Берем до 4 признаков для лучшей визуализации
    if not numeric_cols:
        return

    subset = df[numeric_cols].dropna()
    if subset.empty:
        return

    melted = subset.melt(var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="feature", y="value", data=melted, ax=ax)
    ax.set_title("Outlier Detection using Box Plots")
    ax.set_ylabel("Value")
    ax.set_xlabel("Features")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "outlier_boxplots.png", bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(class_df: pd.DataFrame, reg_df: pd.DataFrame):
    """Визуализирует влияние трансформаций выбросов на accuracy и RMSE."""
    acc_by_outlier = (
        class_df.groupby("Outlier")["Accuracy"].mean().reindex(OUTLIER_METHODS)
    )
    rmse_by_outlier = reg_df.groupby("Outlier")["RMSE"].mean().reindex(OUTLIER_METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    acc_by_outlier.plot.bar(ax=axes[0], color="tab:green", edgecolor="black")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Classification accuracy by outlier transformation")
    axes[0].tick_params(axis='x', rotation=45)
    rmse_by_outlier.plot.bar(ax=axes[1], color="tab:orange", edgecolor="black")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Regression RMSE by outlier transformation")
    axes[1].tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "outlier_metric_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, config: dict):
    """Сохраняет тепловую карту корреляций числовых признаков."""
    numeric = df[config["classification_numeric"]].copy()
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Numeric Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numeric_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_target_boxplot(df: pd.DataFrame, config: dict):
    """Сохраняет боксплоты ключевых признаков, разделенных по целевому признаку."""
    target = config["classification_target"]
    numeric_cols = config["classification_numeric"][:2]
    if not numeric_cols:
        return
    subset = df[[target] + numeric_cols].dropna()
    melted = subset.melt(id_vars=target, var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="feature", y="value", hue=target, data=melted, palette="Set2", ax=ax)
    ax.set_title(f"Feature distributions by {target}")
    ax.legend(title=target)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_survival_boxplot.png", bbox_inches="tight")
    plt.close(fig)


def main(dataset_id=None, max_rows=None, use_gpu=False):
    dataset_id = dataset_id or DEFAULT_DATASET_ID
    config = get_dataset_config(dataset_id)
    df = load_dataset(config, max_rows=max_rows)
    print(f"\nИспользуем датасет {config['name']} (ID {dataset_id}): {config['description']}\n")

    info = f"{df.shape[0]} rows × {df.shape[1]} cols"
    if max_rows:
        info += f" (limited to {max_rows})"
    log_stage("Data load", "DONE", info)

    if use_gpu:
        if CuLogisticRegression is None or CuRandomForestRegressor is None:
            log_stage("GPU mode", "FAILED", "cuML not available, using CPU estimators")
            use_gpu = False
        else:
            log_stage("GPU mode", "ENABLED", "cuML estimators will be used")

    classification_results = []
    classification_combos = list(product(OUTLIER_METHODS, MISSING_STRATEGIES, SCALING_METHODS))
    log_stage("Classification", "START", f"{len(classification_combos)} combos")
    for outlier, imputer, scaler in tqdm(classification_combos, desc="Classification", unit="combo", leave=False):
        try:
            record = evaluate_classification(df, config, outlier, imputer, scaler, use_gpu=use_gpu)
            record["Dataset"] = config["name"]
            classification_results.append(record)
        except Exception as e:
            continue
    log_stage("Classification", "DONE")

    classification_df = pd.DataFrame(classification_results)
    classification_table = (
        classification_df
        .sort_values("Accuracy", ascending=False)
        .round(
            {
                "Accuracy": 3,
                "Precision": 3,
                "Recall": 3,
                "F1": 3,
                "SampleCount": 0,
                "FoldCount": 0,
            }
        )
    )
    print("\nTop classification pipelines (higher accuracy is better):")
    print(classification_table.head(5).to_string(index=False))

    regression_data = df[config["regression_features"] + [config["regression_target"]]].dropna(
        subset=[config["regression_target"]]
    )
    regression_results = []
    regression_combos = list(product(OUTLIER_METHODS, MISSING_STRATEGIES, SCALING_METHODS))
    log_stage("Regression", "START", f"{len(regression_combos)} combos")
    for outlier, imputer, scaler in tqdm(regression_combos, desc="Regression", unit="combo", leave=False):
        try:
            record = evaluate_regression(regression_data, config, outlier, imputer, scaler, use_gpu=use_gpu)
            record["Dataset"] = config["name"]
            regression_results.append(record)
        except Exception as e:
            continue
    log_stage("Regression", "DONE")

    regression_df = pd.DataFrame(regression_results)
    regression_table = (
        regression_df
        .sort_values("RMSE", ascending=True)
        .round({"RMSE": 3, "MSE": 3, "MAE": 3, "R2": 3, "SampleCount": 0, "FoldCount": 0})
    )
    print("\nTop regression pipelines (lower RMSE is better):")
    print(regression_table.head(5).to_string(index=False))

    ensure_results_dir()
    class_path = RESULTS_DIR / f"classification_full_{dataset_id}.csv"
    reg_path = RESULTS_DIR / f"regression_full_{dataset_id}.csv"
    classification_df.to_csv(class_path, index=False)
    regression_df.to_csv(reg_path, index=False)
    log_stage("Saving results", "DONE", f"{class_path.name}, {reg_path.name}")

    drift_df = df.dropna(subset=[config["classification_target"]])
    allowed = config.get("classification_allowed_values")
    if allowed:
        drift_df = drift_df[drift_df[config["classification_target"]].isin(allowed)]
    if drift_df.shape[0] < 2 or drift_df[config["classification_target"]].nunique() < 2:
        print("\nНедостаточно данных для определения дрейфа.")
        drift_report = pd.DataFrame(
            [{"feature": col, "p_value": np.nan, "drift": False, "note": "not enough samples"} for col in config["classification_numeric"]]
        )
        drift_detected = False
    else:
        reference, current = train_test_split(
            drift_df,
            test_size=0.5,
            stratify=drift_df[config["classification_target"]],
            random_state=42,
        )
        drift_report = detect_drift(reference, current, config["classification_numeric"])
        drift_detected = drift_report["drift"].any()
    print("\nData drift diagnostics (KS test per numeric feature):")
    print(drift_report.to_string(index=False))
    print(
        f"\nDrift detected: {'Yes (p < 0.05)' if drift_detected else 'No (all p >= 0.05)'} for at least one numeric feature."
    )

    best_classification = classification_table.iloc[0]
    print(
        "\nBest classification strategy:"
        f" Imputer={best_classification['Imputer']}, "
        f"Outlier={best_classification['Outlier']}, "
        f"Scaler={best_classification['Scaler']} => Accuracy {best_classification['Accuracy']}"
    )

    # Вычисление confusion matrix для лучшей модели классификации
    best_outlier = best_classification['Outlier']
    best_imputer = best_classification['Imputer']
    best_scaler = best_classification['Scaler']

    prepared = apply_drop_strategy(df, config["classification_numeric"] + config["classification_categorical"], [config["classification_target"]], best_imputer)
    filtered = detect_outliers(prepared, config["classification_numeric"], method=best_outlier)
    filtered = filtered.dropna(subset=[config["classification_target"]])
    allowed = config.get("classification_allowed_values")
    if allowed:
        filtered = filtered[filtered[config["classification_target"]].isin(allowed)]

    X_best = filtered[config["classification_features"]]
    y_best = filtered[config["classification_target"]]

    numeric_imputer = get_numeric_imputer(best_imputer)
    scaler = get_scaler(best_scaler)
    pipeline_best = build_preprocessing_pipeline(
        config["classification_numeric"],
        config["classification_categorical"],
        numeric_imputer,
        scaler,
        build_classification_estimator(use_gpu),
    )

    # Обучаем на всех данных и получаем предсказания
    pipeline_best.fit(X_best, y_best)
    y_pred = pipeline_best.predict(X_best)

    cm = confusion_matrix(y_best, y_pred)
    print("\nConfusion Matrix for best classification model:")
    print(cm)
    best_regression = regression_table.iloc[0]
    print(
        "\nBest regression strategy:"
        f" Imputer={best_regression['Imputer']}, "
        f"Outlier={best_regression['Outlier']}, "
        f"Scaler={best_regression['Scaler']} => RMSE {best_regression['RMSE']}\n"
    )

    ensure_figures_dir()
    plot_missingness(df, config)
    plot_outlier_distributions(df, config)
    plot_metric_comparison(classification_df, regression_df)
    plot_correlation_heatmap(df, config)
    plot_target_boxplot(df, config)
    print("\nВизуализации сохранены в папку figures/")


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset, max_rows=args.max_rows, use_gpu=args.use_gpu)
