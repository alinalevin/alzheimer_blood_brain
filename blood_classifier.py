"""Offline version of `Copy_of_blood_ad_classifier.ipynb` with reusable
functions per classification task (AD vs Control, MCI vs Control).
The script relies on helper utilities from `classification_eval.py` and can be
invoked as a command-line tool or consumed as a module.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import os
import sys
from contextlib import contextmanager
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import make_scorer, matthews_corrcoef

from classification_eval import (
    eval_cls_model,
    feature_selection_elastic_net,
    get_ad_control_train_test_data,
    get_mci_control_train_test_data,
    get_rf_feature_importance,
    plot_all_models_roc_curve,
    plot_lr_feature_importance,
    plot_rf_feature_importance,
    plot_xgb_feature_importance,
    plot_roc_auc_for_model,
    print_test_scores,
    print_train_scores,
)


class TaskTee:
    def __init__(self, task_name: str, log_dir: Path):
        date_str = datetime.now().strftime("%Y%m%d")
        safe_task = task_name.lower()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"blood_{safe_task}_{date_str}.log"

        # Store original stdout/stderr before any modifications
        self._stdout = sys.stdout
        self._stderr = sys.stderr

        # Clear terminal (platform-specific)
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/Mac
            os.system('clear')

        # Clear the log file by opening in write mode first
        with self.log_path.open("w", encoding="utf-8") as f:
            f.write("")  # Clear existing content

        # Now open in append mode for logging
        self._log_file = self.log_path.open("a", encoding="utf-8")

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._log_file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._log_file.flush()

    def close(self) -> None:
        self._log_file.close()


@contextmanager
def task_logging(task_name: str):
    log_dir = Path(__file__).resolve().parent / "files"
    tee = TaskTee(task_name, log_dir)
    sys.stdout = sys.stderr = tee
    try:
        yield tee.log_path
    finally:
        sys.stdout = tee._stdout
        sys.stderr = tee._stderr
        tee.close()


BASE_SCORERS = {
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "matthews_corrcoef": make_scorer(matthews_corrcoef),
}


def select_scorers(metric_names: List[str]) -> Dict[str, object]:
    return {name: BASE_SCORERS[name] for name in metric_names}


def describe_dataset(tissue_name: str, counts: pd.DataFrame, train_metadata: pd.DataFrame, test_metadata: pd.DataFrame) -> None:
    """Print summary statistics about the dataset."""
    print(f"\n{'='*60}")
    print(f"{tissue_name} Dataset Summary")
    print(f"{'='*60}")
    print(f"Total genes: {counts.shape[0]}")
    print(f"Total samples: {counts.shape[1]}")
    print(f"Train samples: {len(train_metadata)}")
    print(f"Test samples: {len(test_metadata)}")
    print(f"{'='*60}\n")


@dataclass
class BloodClassifierConfig:
    metadata_path: Path
    counts_path: Path
    train_metadata_path: Path
    test_metadata_path: Path
    ad_gene_path: Path
    mci_gene_path: Path
    outlier_ids: List[str]
    feature_alpha: float = 0.001
    feature_l1_ratio: float = 0.05
    tissue_name: str = "Blood"
    output_dir: Optional[Path] = None

    @staticmethod
    def default(out_dir: Optional[Path] = None) -> "BloodClassifierConfig":
        data_dir = Path(__file__).resolve().parent / "files" / "blood"
        if not data_dir.exists():
            raise FileNotFoundError(f"Expected data directory at {data_dir}")
        return BloodClassifierConfig(
            metadata_path=data_dir / "blood_final_residuals_metadata_all.xlsx",
            counts_path=data_dir / "blood_r1_clean_all.csv",
            train_metadata_path=data_dir / "blood_train_metadata_all.xlsx",
            test_metadata_path=data_dir / "blood_test_metadata_all.xlsx",
            ad_gene_path=data_dir / "dge_meta_results_ad_control.xlsx",
            mci_gene_path=data_dir / "dge_meta_results_mci_control.xlsx",
            outlier_ids=["sample_053", "sample_293", "sample_588", "sample_286"],
            output_dir=out_dir,
        )


def load_metadata(path: Path, drop_ids: List[str]) -> pd.DataFrame:
    metadata = pd.read_excel(path)
    specimen_col = "specimenID" if "specimenID" in metadata.columns else None
    index_col = "individualID" if "individualID" in metadata.columns else specimen_col or metadata.columns[0]
    if drop_ids and specimen_col:
        metadata = metadata[~metadata[specimen_col].isin(drop_ids)]
    metadata.set_index(index_col, inplace=True, drop=False)
    return metadata


def load_counts(path: Path, specimen_ids: List[str]) -> pd.DataFrame:
    counts = pd.read_csv(path)
    first_col = counts.columns[0]
    if first_col != "ensembl_gene_id":
        counts.rename(columns={first_col: "ensembl_gene_id"}, inplace=True)
    counts.set_index("ensembl_gene_id", inplace=True, drop=True)
    keep_cols = [col for col in specimen_ids if col in counts.columns]
    return counts[keep_cols]


def load_gene_frame(path: Path) -> pd.DataFrame:
    gene_df = pd.read_excel(path)
    gene_df.set_index("ensembl_gene_id", inplace=True, drop=True)
    return gene_df


def _select_feature_names(feature_df: pd.DataFrame) -> List[str]:
    if "ensemble_gene_id" in feature_df.columns:
        return feature_df["ensemble_gene_id"].tolist()
    if "ensembl_gene_id" in feature_df.columns:
        return feature_df["ensembl_gene_id"].tolist()
    raise ValueError("Feature importance output missing gene id column.")


class TaskData:
    def __init__(
        self,
        name: str,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_test: pd.DataFrame,
        y_test: np.ndarray,
        genes_df: pd.DataFrame,
        feature_names: List[str],
    ) -> None:
        self.name = name
        self.x_train = x_train[feature_names]
        self.y_train = y_train
        self.x_test = x_test[feature_names]
        self.y_test = y_test
        self.genes_df = genes_df
        self.feature_names = feature_names
        self.class_names = ["CONTROL", "AD"] if name == "AD_CONTROL" else ["CONTROL", "MCI"]


TASK_SPLITTERS = {
    "AD_CONTROL": (get_ad_control_train_test_data, "MCI"),
    "MCI_CONTROL": (get_mci_control_train_test_data, "AD"),
}


def prepare_task_data(
    task_name: str,
    counts: pd.DataFrame,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    genes_df: pd.DataFrame,
    alpha: float,
    l1_ratio: float,
) -> TaskData:
    splitter, remove_condition = TASK_SPLITTERS[task_name]
    x_train, y_train, x_test, y_test = splitter(
        counts,
        train_metadata,
        test_metadata,
        genes_df,
        condition_to_remove=remove_condition,
    )
    feature_df = feature_selection_elastic_net(x_train, y_train, alpha=alpha, l1_ratio=l1_ratio)
    selected = _select_feature_names(feature_df)
    return TaskData(task_name, x_train, y_train, x_test, y_test, genes_df, selected)


def compute_scale_pos_weight(y: np.ndarray) -> float:
    values, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(values, counts))
    negatives = class_counts.get(0, 0)
    positives = class_counts.get(1, 1)
    return float(negatives) / float(positives) if positives else 1.0


def train_balanced_random_forest(task_data: TaskData) -> BalancedRandomForestClassifier:
    rf_params = {
        "n_estimators": 300 if task_data.name == "AD_CONTROL" else 150,
        "criterion": "entropy",
        "random_state": 42,
        "sampling_strategy": "not minority" if task_data.name == "AD_CONTROL" else "all",
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "bootstrap": False,
        "n_jobs": 8,
    }
    if task_data.name == "AD_CONTROL":
        rf_params["min_samples_split"] = 5
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    metrics = ["f1", "recall", "precision", "roc_auc", "matthews_corrcoef", "accuracy"]
    scores = cross_validate(
        BalancedRandomForestClassifier(**rf_params),
        task_data.x_train,
        task_data.y_train,
        scoring=select_scorers(metrics),
        cv=cv,
        return_train_score=True,
        n_jobs=rf_params["n_jobs"],
    )
    print_train_scores(scores)
    print_test_scores(scores)
    model = BalancedRandomForestClassifier(**rf_params)
    sample_weight = compute_sample_weight("balanced", y=task_data.y_train)
    model.fit(task_data.x_train, task_data.y_train, sample_weight=sample_weight)
    eval_cls_model(
        model,
        task_data.x_test,
        task_data.y_test,
        labels=task_data.class_names,
        model_name="Balanced Random Forest",
        task_name=task_data.name
    )
    plot_roc_auc_for_model(model, "Balanced Random Forest", task_data.x_test, task_data.y_test, model_type=task_data.name)
    importance_df = get_rf_feature_importance(model, task_data.x_train, task_data.genes_df, task_data.name)
    plot_rf_feature_importance(importance_df, task_data.name)
    return model


def train_xgboost(task_data: TaskData) -> xgb.XGBClassifier:
    xgb_params = {
        "n_estimators": 150,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "learning_rate": 0.2 if task_data.name == "AD_CONTROL" else 0.25,
        "max_depth": 4 if task_data.name == "AD_CONTROL" else 3,
        "min_child_weight": 1,
        "colsample_bytree": 0.1 if task_data.name == "AD_CONTROL" else 0.2,
        "n_jobs": 8,
        "scale_pos_weight": compute_scale_pos_weight(task_data.y_train),
    }
    model = xgb.XGBClassifier(**xgb_params)
    # Note: Skipping cross-validation for XGBoost due to sklearn compatibility issues
    # The model will be trained directly on the full training set
    weights = compute_sample_weight("balanced", y=task_data.y_train)
    model.fit(task_data.x_train, task_data.y_train, sample_weight=weights)
    eval_cls_model(
        model,
        task_data.x_test,
        task_data.y_test,
        labels=task_data.class_names,
        model_name="XGBoost",
        task_name=task_data.name
    )
    plot_roc_auc_for_model(model, "XGBoost", task_data.x_test, task_data.y_test, model_type=task_data.name)
    plot_xgb_feature_importance(
        model,
        task_data.name,
        task_data.genes_df.reset_index(drop=False, inplace=False),
        "blood",
    )
    return model


def train_logistic_regression(task_data: TaskData) -> Pipeline:
    lr = LogisticRegression(
        random_state=42,
        fit_intercept=True,
        class_weight="balanced",
        max_iter=300,
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.1,
        C=0.1,
        n_jobs=8,
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", lr)])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    metrics = ["f1", "recall", "precision", "roc_auc", "matthews_corrcoef", "accuracy"]
    scores = cross_validate(
        pipeline,
        task_data.x_train,
        task_data.y_train,
        scoring=select_scorers(metrics),
        cv=cv,
    )
    print_train_scores(scores)
    print_test_scores(scores)
    pipeline.fit(task_data.x_train, task_data.y_train)
    eval_cls_model(
        pipeline,
        task_data.x_test,
        task_data.y_test,
        labels=task_data.class_names,
        model_name="Logistic Regression",
        task_name=task_data.name
    )
    plot_roc_auc_for_model(pipeline, "Logistic Regression", task_data.x_test, task_data.y_test, model_type=task_data.name)
    plot_lr_feature_importance(
        pipeline.named_steps["clf"],
        task_data.name,
        task_data.x_train,
        task_data.genes_df,
        "blood",
    )
    return pipeline


def compare_models(task_data: TaskData, models: Dict[str, object], output_dir: Optional[Path], tissue: str) -> None:
    if not output_dir:
        output_dir = Path.cwd() / "blood_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_pred_probs = {name: est.predict_proba(task_data.x_test)[:, 1] for name, est in models.items()}
    plot_all_models_roc_curve(task_data.y_test, model_pred_probs, task_data.name, tissue, str(output_dir))


def persist_model(model, model_name: str, model_type: str, task_data: TaskData, output_dir: Optional[Path]) -> None:
    if not output_dir:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}_{model_type}.pkl"
    joblib.dump(model, model_path)
    task_data.x_train.to_csv(output_dir / f"x_train_{model_name}_{model_type}.csv")
    task_data.x_test.to_csv(output_dir / f"x_test_{model_name}_{model_type}.csv")
    y_train = pd.DataFrame(task_data.y_train, index=task_data.x_train.index, columns=["condition"])
    y_test = pd.DataFrame(task_data.y_test, index=task_data.x_test.index, columns=["condition"])
    y_train.to_csv(output_dir / f"y_train_{model_name}_{model_type}.csv")
    y_test.to_csv(output_dir / f"y_test_{model_name}_{model_type}.csv")


def run_task(
    task_name: str,
    counts: pd.DataFrame,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    genes_df: pd.DataFrame,
    config: BloodClassifierConfig,
) -> Dict[str, object]:
    if config.output_dir:
        os.environ["BLOOD_OUTPUT_DIR"] = str(config.output_dir)
    with task_logging(task_name):
        print(f"\n=== Running {task_name} Blood classification ===")
        task_data = prepare_task_data(
            task_name=task_name,
            counts=counts,
            train_metadata=train_metadata,
            test_metadata=test_metadata,
            genes_df=genes_df,
            alpha=config.feature_alpha,
            l1_ratio=config.feature_l1_ratio,
        )
        print(f"Training Balanced Random Forest for {task_name}...")
        rf_model = train_balanced_random_forest(task_data)
        print(f"Training XGBoost for {task_name}...")
        xgb_model = train_xgboost(task_data)
        print(f"Training Logistic Regression for {task_name}...")
        lr_model = train_logistic_regression(task_data)
        print(f"Completed {task_name} classification. Comparing models...")
        compare_models(task_data, {"RF": rf_model, "XGB": xgb_model, "LR": lr_model}, config.output_dir, config.tissue_name)
        persist_model(rf_model, "rf_blood", task_name, task_data, config.output_dir or Path.cwd())
        persist_model(xgb_model, "xgb_blood", task_name, task_data, config.output_dir or Path.cwd())
        persist_model(lr_model, "lr_blood", task_name, task_data, config.output_dir or Path.cwd())
        return {"rf": rf_model, "xgb": xgb_model, "lr": lr_model}


def run_pipeline(config: BloodClassifierConfig, tasks: Optional[List[str]] = None) -> Dict[str, Dict[str, object]]:
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        os.environ["BLOOD_OUTPUT_DIR"] = str(config.output_dir)
    master_metadata = load_metadata(config.metadata_path, config.outlier_ids)
    train_metadata = load_metadata(config.train_metadata_path, config.outlier_ids)
    test_metadata = load_metadata(config.test_metadata_path, config.outlier_ids)
    specimen_ids = master_metadata["specimenID"].tolist()
    counts = load_counts(config.counts_path, specimen_ids)
    if "individualID" in master_metadata.columns:
        specimen_to_individual_map = master_metadata.dropna(subset=["specimenID", "individualID"]).set_index("specimenID")["individualID"]
        counts = counts.rename(columns=specimen_to_individual_map)
        if counts.columns.duplicated().any():
            counts = counts.T.groupby(level=0).first().T
    describe_dataset("Blood", counts, train_metadata, test_metadata)
    genes_ad = load_gene_frame(config.ad_gene_path)
    genes_mci = load_gene_frame(config.mci_gene_path)
    results = {}
    selected_tasks = tasks or ["AD_CONTROL", "MCI_CONTROL"]
    if "AD_CONTROL" in selected_tasks:
        task = run_task("AD_CONTROL", counts, train_metadata, test_metadata, genes_ad, config)
        results["AD_CONTROL"] = task
    if "MCI_CONTROL" in selected_tasks:
        results["MCI_CONTROL"] = run_task("MCI_CONTROL", counts, train_metadata, test_metadata, genes_mci, config)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood classification pipeline")
    parser.add_argument("--metadata", type=Path, default=None, help="Path to full metadata excel file")
    parser.add_argument("--train-metadata", type=Path, default=None, help="Path to train metadata excel file")
    parser.add_argument("--test-metadata", type=Path, default=None, help="Path to test metadata excel file")
    parser.add_argument("--counts", type=Path, default=None, help="Path to counts CSV file")
    parser.add_argument("--ad-genes", type=Path, default=None, help="Path to AD vs Control genes excel file")
    parser.add_argument("--mci-genes", type=Path, default=None, help="Path to MCI vs Control genes excel file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to store trained models and csvs")
    parser.add_argument("--tasks", nargs="*", choices=["AD_CONTROL", "MCI_CONTROL"], help="Subset of tasks to run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BloodClassifierConfig.default(args.output_dir)
    if args.metadata:
        config.metadata_path = args.metadata
    if args.train_metadata:
        config.train_metadata_path = args.train_metadata
    if args.test_metadata:
        config.test_metadata_path = args.test_metadata
    if args.counts:
        config.counts_path = args.counts
    if args.ad_genes:
        config.ad_gene_path = args.ad_genes
    if args.mci_genes:
        config.mci_gene_path = args.mci_genes
    if args.output_dir:
        config.output_dir = args.output_dir
        os.environ["BLOOD_OUTPUT_DIR"] = str(args.output_dir)
    run_pipeline(config, args.tasks)


if __name__ == "__main__":
    main()
