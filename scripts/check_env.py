from __future__ import annotations

import importlib
import sys


MODULES = {
    "core": [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "networkx",
    ],
    "imbalanced_data": [
        "imblearn",
        "xgboost",
        "lightgbm",
        "optuna",
    ],
    "interpretability": [
        "lime",
        "shap",
    ],
    "graphs_and_node2vec": [
        "gensim",
        "node2vec",
    ],
    "gnn": [
        "torch",
        "torch_geometric",
    ],
    "notebooks": [
        "IPython",
        "ipykernel",
        "jupyterlab",
    ],
}


def check_import(module_name: str) -> tuple[str, bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - diagnostics script should show import failures.
        return module_name, False, repr(exc)

    version = getattr(module, "__version__", "version unknown")
    return module_name, True, str(version)


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    failed = []

    for group, modules in MODULES.items():
        print(f"\n[{group}]")
        for module_name in modules:
            name, ok, detail = check_import(module_name)
            status = "OK" if ok else "FAIL"
            print(f"{status:4} {name:16} {detail}")
            if not ok:
                failed.append(name)

    if failed:
        print("\nMissing or broken modules:", ", ".join(failed))
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
