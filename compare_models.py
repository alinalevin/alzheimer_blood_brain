"""Utility helpers for comparing blood and brain classifiers on shared samples."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import re

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "files"
LOG_DIR = DEFAULT_RESULTS_DIR
DEFAULT_METADATA_PATH = LOG_DIR / "blood" / "blood_final_residuals_metadata_all.xlsx"

LOG_PATTERNS: Dict[str, Dict[str, str]] = {
    "blood": {
        "AD_CONTROL": "blood_ad_control",
        "MCI_CONTROL": "blood_mci_control",
    },
    "brain": {
        "AD_CONTROL": "brain_ad_control",
        "MCI_CONTROL": "brain_mci_control",
    },
}

MODEL_ALIASES = {
    "Balanced Random Forest": "Balanced Random Forest",
    "XGBoost": "XGBoost",
    "Logistic Regression": "Logistic Regression",
}
EXPECTED_MODELS = tuple(MODEL_ALIASES.values())

METRIC_PREFIXES = {
    "accuracy": "Accuracy:",
    "recall": "Sensitivity (recall)",
    "precision": "Precision Score",
    "specificity": "Specificity",
    "roc_auc": "ROC AUC score",
    "f1": "F1 Score",
    "mcc": "Matthews correlation coefficient",
}
METRIC_ORDER = ["accuracy", "precision", "recall", "specificity", "roc_auc", "f1", "mcc"]
GROUP_ORDER = ("ALL", "CONTROL", "AD", "MCI")
TASK_LABELS = {
    "AD_CONTROL": {"CONTROL": 0, "AD": 1},
    "MCI_CONTROL": {"CONTROL": 0, "MCI": 1},
}

MODEL_CHOICES = ("rf", "xgb", "lr")
DEFAULT_MODEL = "rf"
MODEL_CODE_TO_NAME = {
    "rf": "Balanced Random Forest",
    "xgb": "XGBoost",
    "lr": "Logistic Regression",
}


def build_task_result_files(model: str = DEFAULT_MODEL, file_suffix: str = "") -> Dict[str, Path]:
    """Return task result file paths for a given model suffix.

    Supports both old filenames (no model suffix, e.g. comparison_results_AD_CONTROL_DLPFC.csv)
    and new filenames (with model suffix, e.g. comparison_results_AD_CONTROL_DLPFC_rf.csv).
    The model-suffixed file takes precedence when it exists.
    """
    tasks = {
        "AD_CONTROL_DLPFC": "comparison_results_AD_CONTROL_DLPFC",
        "AD_CONTROL_PCC": "comparison_results_AD_CONTROL_PCC",
        "MCI_CONTROL_DLPFC": "comparison_results_MCI_CONTROL_DLPFC",
        "MCI_CONTROL_PCC": "comparison_results_MCI_CONTROL_PCC",
    }
    result = {}
    for key, base in tasks.items():
        with_model = LOG_DIR / f"{base}_{model}{file_suffix}.csv"
        without_model = LOG_DIR / f"{base}{file_suffix}.csv"
        legacy_with_model = LOG_DIR / f"{base}_{model}.csv"
        legacy_without_model = LOG_DIR / f"{base}.csv"
        if with_model.exists():
            result[key] = with_model
        elif without_model.exists():
            result[key] = without_model
        elif not file_suffix and legacy_with_model.exists():
            result[key] = legacy_with_model
        elif not file_suffix:
            result[key] = legacy_without_model
        else:
            result[key] = without_model
    return result


# Default (backward-compatible)
TASK_RESULT_FILES = build_task_result_files(DEFAULT_MODEL)

TASK_METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

TASK_CORRECT_COLS = {
    "blood": "blood_correct",
    "brain": "brain_correct",
}

BETTER_MODEL_ORDER = ["blood", "brain", "both", "neither"]
TISSUE_LABELS = ("DLPFC", "PCC")


def _log_stamp(log_path: Path) -> str:
    """Extract sortable timestamp suffix from log filename."""
    match = re.search(r"_(\d{8}(?:_\d{6})?)$", log_path.stem)
    return match.group(1) if match else ""


def _find_log_candidates(log_dir: Path, prefix: str) -> List[Path]:
    """Return matching logs, newest first, based on filename timestamp."""
    candidates = list(log_dir.glob(f"{prefix}_*.log"))
    candidates.sort(key=lambda p: (_log_stamp(p), p.name), reverse=True)
    if not candidates:
        print(f"No log file found matching pattern: {prefix}_*.log")
    return candidates


def _line_value(line: str) -> Optional[float]:
    try:
        return float(line.split(":", 1)[-1].strip())
    except ValueError:
        return None


def _parse_log_file(log_path: Path, tissue: str, expected_task: str) -> pd.DataFrame:
    training_re = re.compile(r"Training (.+?) for (AD_CONTROL|MCI_CONTROL)\.")
    records: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    skip_current = False
    seen_keys = set()

    def commit() -> None:
        nonlocal current
        if current and not skip_current:
            missing = [k for k in METRIC_ORDER if k not in current]
            if missing:
                # Debug: show what metrics are missing
                model = current.get("model", "Unknown")
                task = current.get("task", "Unknown")
                print(f"  Skipping incomplete {tissue} {model} {task} (missing: {', '.join(missing)})")
                current = None
                return
            # Debug: show successful parse
            model = current.get("model", "Unknown")
            task = current.get("task", "Unknown")
            print(f"  + Parsed {tissue} {model} {task}")
            records.append(current)
        current = None

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = training_re.search(line)
            if match:
                commit()
                model_name, task = match.groups()
                if task != expected_task:
                    skip_current = True
                    current = None
                    continue
                model = MODEL_ALIASES.get(model_name, model_name)
                key = (tissue, task, model)
                skip_current = key in seen_keys
                if not skip_current:
                    current = {
                        "tissue": tissue,
                        "task": task,
                        "model": model,
                        "log_path": str(log_path),
                    }
                    seen_keys.add(key)
                else:
                    current = None
                continue
            if skip_current or current is None:
                continue
            for metric, prefix in METRIC_PREFIXES.items():
                if line.startswith(prefix):
                    value = _line_value(line)
                    if value is not None:
                        current[metric] = value
                    break
    commit()
    return pd.DataFrame.from_records(records)


def build_log_comparison(log_dir: Path = LOG_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Searching for log files in: {log_dir} ===")
    frames: List[pd.DataFrame] = []
    for tissue, tasks in LOG_PATTERNS.items():
        for task, prefix in tasks.items():
            candidates = _find_log_candidates(log_dir, prefix)
            if not candidates:
                continue
            best_by_model: Dict[str, Dict[str, object]] = {}
            for log_path in candidates:
                if len(best_by_model) == len(EXPECTED_MODELS):
                    break
                print(f"Checking log file: {log_path}")
                df = _parse_log_file(log_path, tissue, task)
                if df.empty:
                    continue
                for row in df.to_dict("records"):
                    model_name = str(row.get("model", ""))
                    if model_name and model_name not in best_by_model:
                        best_by_model[model_name] = row
            if best_by_model:
                missing = [model for model in EXPECTED_MODELS if model not in best_by_model]
                if missing:
                    print(f"  Missing log metrics for {tissue} {task}: {', '.join(missing)}")
                frames.append(pd.DataFrame.from_records(list(best_by_model.values())))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    pivot = raw.pivot_table(index=["task", "model"], columns="tissue", values=METRIC_ORDER)
    pivot = pivot.sort_index()
    pivot.columns = [f"{tissue}_{metric}" for metric, tissue in pivot.columns]
    return raw, pivot.reset_index()


def export_log_tables(raw_df: pd.DataFrame, comparison_df: pd.DataFrame, raw_path: Path, comparison_path: Path) -> None:
    if not raw_df.empty:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_path, index=False, float_format="%.4f")
        print(f"Saved log metrics to {raw_path}")
    if not comparison_df.empty:
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(comparison_path, index=False, float_format="%.4f")
        print(f"Saved log comparison table to {comparison_path}")


def _load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Comparison file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "true_label_num" not in df.columns:
        raise ValueError("CSV must include 'true_label_num' column")
    return df


def _compute_metric_suite(df: pd.DataFrame, prediction_col: str, proba_col: str) -> Dict[str, float]:
    y_true = df["true_label_num"].astype(int)
    y_pred = df[prediction_col].astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if proba_col in df:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, df[proba_col].astype(float)))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def _compute_full_metric_suite(df: pd.DataFrame, prediction_col: str, proba_col: str) -> Dict[str, float]:
    y_true = df["true_label_num"].astype(int)
    y_pred = df[prediction_col].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, df[proba_col].astype(float)))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _empty_metric_suite() -> Dict[str, float]:
    return {metric: float("nan") for metric in METRIC_ORDER}


def _group_mask(df: pd.DataFrame, group: str, task_label: str) -> pd.Series:
    if group == "ALL":
        return pd.Series(True, index=df.index)

    if "true_label_blood" in df.columns:
        labels = df["true_label_blood"].astype(str).str.upper()
    elif "true_label" in df.columns:
        labels = df["true_label"].astype(str).str.upper()
    elif "true_label_num" in df.columns:
        inverse = {v: k for k, v in TASK_LABELS[task_label].items()}
        labels = df["true_label_num"].astype(int).map(inverse).fillna("").astype(str).str.upper()
    else:
        return pd.Series(False, index=df.index)

    return labels == group


def _compute_task_metrics(df: pd.DataFrame, prediction_col: str, proba_col: str) -> Dict[str, float]:
    y_true = df["true_label_num"].astype(int)
    y_pred = df[prediction_col].astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, df[proba_col].astype(float))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _compute_task_breakdown(df: pd.DataFrame) -> Dict[str, int]:
    counts = df["better_model"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in BETTER_MODEL_ORDER}


def summarize_better_model(df: pd.DataFrame) -> Dict[str, int]:
    counts = df.get("better_model")
    if counts is None:
        return {label: 0 for label in BETTER_MODEL_ORDER}
    value_counts = df["better_model"].value_counts().to_dict()
    return {label: int(value_counts.get(label, 0)) for label in BETTER_MODEL_ORDER}


def _ensure_individual_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "individualID" in df.columns:
        return df
    first_col = df.columns[0]
    if first_col != "individualID":
        df = df.rename(columns={first_col: "individualID"})
    return df


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.lower()
    mapped = lowered.map({"true": True, "false": False})
    if mapped.isnull().all():
        return series.astype(bool)
    return mapped.fillna(False)


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_excel(metadata_path)
    if "individualID" not in metadata.columns or "cogdx" not in metadata.columns:
        raise ValueError(f"Metadata file {metadata_path} missing 'individualID' or 'cogdx' columns")
    subset = metadata[["individualID", "cogdx"]].dropna(subset=["individualID"])
    return subset.drop_duplicates(subset=["individualID"])


def analyze_cogdx_failures(results_df: pd.DataFrame, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_df = _ensure_individual_id_column(results_df.copy())
    for col in ("blood_correct", "brain_correct"):
        if col in results_df.columns:
            results_df[col] = _coerce_bool(results_df[col])
        else:
            raise KeyError(f"Expected column '{col}' in results dataframe")
    merged = results_df.merge(metadata, on="individualID", how="left")
    failure_mask = (~merged["blood_correct"]) & (~merged["brain_correct"])
    failures = merged[failure_mask].copy()
    failure_counts = failures.groupby("cogdx")["individualID"].count().rename("failed_count")
    total_counts = merged.groupby("cogdx")["individualID"].count().rename("total_count")
    summary = (
        pd.concat([failure_counts, total_counts], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    summary["failure_rate"] = summary.apply(
        lambda row: row["failed_count"] / row["total_count"] if row["total_count"] else 0,
        axis=1,
    )
    summary = summary.sort_values(by="failure_rate", ascending=False)
    return summary, failures[["individualID", "task", "cogdx", "blood_correct", "brain_correct"]]


def compare_models(
    csv_path: Optional[Path] = None,
    task_files: Optional[Dict[str, Path]] = None,
) -> tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    selected_task_files = task_files or TASK_RESULT_FILES
    if csv_path is not None:
        try:
            df = _load_data(csv_path)
        except FileNotFoundError:
            print(f"Overall comparison file {csv_path} not found, falling back to task-level CSVs.")
            df = _load_task_datasets(selected_task_files)
    else:
        df = _load_task_datasets(selected_task_files)
    df = _ensure_individual_id_column(df)
    blood_metrics = _compute_metric_suite(df, "prediction_blood", "probability_blood")
    brain_metrics = _compute_metric_suite(df, "prediction_brain", "probability_brain")
    summary = pd.DataFrame([blood_metrics, brain_metrics], index=["blood", "brain"])
    better_counts = summarize_better_model(df)
    print("=== Model Metrics ===")
    for name, row in summary.iterrows():
        print(f"{name.capitalize():>5}: {_format_metrics(row.to_dict())}")
    return summary, better_counts, df


def export_results(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, float_format="%.4f")
    print(f"Saved metrics to {output_path}")


def export_counts(better_counts: Dict[str, int], output_path: Path) -> None:
    if not better_counts:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([better_counts]).to_csv(output_path, index=False)
    print(f"Saved better-model counts to {output_path}")


def _format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items() if pd.notna(value))


def _task_to_tissue(task_name: object) -> Optional[str]:
    if not isinstance(task_name, str):
        return None
    upper = task_name.upper()
    if upper.endswith("_DLPFC"):
        return "DLPFC"
    if upper.endswith("_PCC"):
        return "PCC"
    return None


def build_tissue_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Tuple[str, Dict[str, float]]] = []
    # Keep brain as one overall row across all available tasks.
    rows.append(("brain", _compute_metric_suite(df, "prediction_brain", "probability_brain")))

    if "task" in df.columns:
        with_tissue = df.copy()
        with_tissue["tissue"] = with_tissue["task"].map(_task_to_tissue)
        with_tissue = with_tissue.dropna(subset=["tissue"])
        for tissue in ("PCC", "DLPFC"):
            subset = with_tissue[with_tissue["tissue"] == tissue]
            if subset.empty:
                continue
            rows.append((f"blood_{tissue}", _compute_metric_suite(subset, "prediction_blood", "probability_blood")))

    labels, metrics = zip(*rows)
    return pd.DataFrame(list(metrics), index=list(labels))


def build_task_model_tissue_metrics(file_suffix: str = "") -> pd.DataFrame:
    datasets = ("PCC", "DLPFC")
    task_labels = ("AD_CONTROL", "MCI_CONTROL")
    task_base_names = {
        "AD_CONTROL_PCC": "comparison_results_AD_CONTROL_PCC",
        "AD_CONTROL_DLPFC": "comparison_results_AD_CONTROL_DLPFC",
        "MCI_CONTROL_PCC": "comparison_results_MCI_CONTROL_PCC",
        "MCI_CONTROL_DLPFC": "comparison_results_MCI_CONTROL_DLPFC",
    }
    rows: List[Dict[str, object]] = []

    for model_code in MODEL_CHOICES:
        task_files: Dict[str, Path] = {}
        # For cross-model table, avoid falling back to unsuffixed files for non-default models.
        for key, base in task_base_names.items():
            model_path = LOG_DIR / f"{base}_{model_code}{file_suffix}.csv"
            default_path = LOG_DIR / f"{base}{file_suffix}.csv"
            if model_path.exists():
                task_files[key] = model_path
            elif model_code == DEFAULT_MODEL and default_path.exists():
                task_files[key] = default_path
        model_name = MODEL_CODE_TO_NAME.get(model_code, model_code)
        for task_label in task_labels:
            for dataset in datasets:
                csv_path = task_files.get(f"{task_label}_{dataset}")
                if not csv_path or not csv_path.exists():
                    print(f"Missing file for {model_name} {task_label}_{dataset}; skipping row.")
                    continue
                df = pd.read_csv(csv_path)
                blood_metrics = _compute_full_metric_suite(df, "prediction_blood", "probability_blood")
                brain_metrics = _compute_full_metric_suite(df, "prediction_brain", "probability_brain")

                record: Dict[str, object] = {
                    "task": task_label,
                    "model": model_name,
                    "combined_dataset": f"blood_{dataset}",
                }
                for metric in METRIC_ORDER:
                    record[f"blood_{metric}"] = blood_metrics.get(metric, float("nan"))
                    record[f"brain_{metric}"] = brain_metrics.get(metric, float("nan"))
                rows.append(record)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    task_order = {label: i for i, label in enumerate(task_labels)}
    model_order = {label: i for i, label in enumerate(EXPECTED_MODELS)}
    dataset_order = {f"blood_{label}": i for i, label in enumerate(datasets)}
    table = table.sort_values(
        by=["task", "model", "combined_dataset"],
        key=lambda col: col.map(
            task_order if col.name == "task" else model_order if col.name == "model" else dataset_order
        ),
    ).reset_index(drop=True)
    return table


def build_task_model_tissue_group_metrics(file_suffix: str = "") -> pd.DataFrame:
    datasets = ("PCC", "DLPFC")
    task_labels = ("AD_CONTROL", "MCI_CONTROL")
    task_base_names = {
        "AD_CONTROL_PCC": "comparison_results_AD_CONTROL_PCC",
        "AD_CONTROL_DLPFC": "comparison_results_AD_CONTROL_DLPFC",
        "MCI_CONTROL_PCC": "comparison_results_MCI_CONTROL_PCC",
        "MCI_CONTROL_DLPFC": "comparison_results_MCI_CONTROL_DLPFC",
    }
    rows: List[Dict[str, object]] = []

    for model_code in MODEL_CHOICES:
        task_files: Dict[str, Path] = {}
        for key, base in task_base_names.items():
            model_path = LOG_DIR / f"{base}_{model_code}{file_suffix}.csv"
            default_path = LOG_DIR / f"{base}{file_suffix}.csv"
            if model_path.exists():
                task_files[key] = model_path
            elif model_code == DEFAULT_MODEL and default_path.exists():
                task_files[key] = default_path

        model_name = MODEL_CODE_TO_NAME.get(model_code, model_code)
        for task_label in task_labels:
            for dataset in datasets:
                csv_path = task_files.get(f"{task_label}_{dataset}")
                if not csv_path or not csv_path.exists():
                    print(f"Missing file for {model_name} {task_label}_{dataset}; skipping grouped rows.")
                    continue

                df = pd.read_csv(csv_path)
                for group in GROUP_ORDER:
                    group_df = df[_group_mask(df, group, task_label)].copy()
                    blood_metrics = (
                        _compute_full_metric_suite(group_df, "prediction_blood", "probability_blood")
                        if not group_df.empty else _empty_metric_suite()
                    )
                    brain_metrics = (
                        _compute_full_metric_suite(group_df, "prediction_brain", "probability_brain")
                        if not group_df.empty else _empty_metric_suite()
                    )

                    record: Dict[str, object] = {
                        "task": task_label,
                        "model": model_name,
                        "combined_dataset": f"blood_{dataset}",
                        "group": group,
                    }
                    for metric in METRIC_ORDER:
                        record[f"blood_{metric}"] = blood_metrics.get(metric, float("nan"))
                        record[f"brain_{metric}"] = brain_metrics.get(metric, float("nan"))
                    rows.append(record)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    task_order = {label: i for i, label in enumerate(task_labels)}
    model_order = {label: i for i, label in enumerate(EXPECTED_MODELS)}
    dataset_order = {f"blood_{label}": i for i, label in enumerate(datasets)}
    group_order = {label: i for i, label in enumerate(GROUP_ORDER)}
    table = table.sort_values(
        by=["task", "model", "combined_dataset", "group"],
        key=lambda col: col.map(
            task_order if col.name == "task"
            else model_order if col.name == "model"
            else dataset_order if col.name == "combined_dataset"
            else group_order
        ),
    ).reset_index(drop=True)
    return table


def compare_tasks(task_files: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, float]] = []
    breakdown_rows: List[Dict[str, int]] = []
    missing: List[str] = []
    for task, csv_path in task_files.items():
        if not csv_path.exists():
            missing.append(task)
            continue
        df = pd.read_csv(csv_path)
        blood = _compute_task_metrics(df, "prediction_blood", "probability_blood")
        brain = _compute_task_metrics(df, "prediction_brain", "probability_brain")
        metric_rows.append(
            {
                "task": task,
                **{f"blood_{k}": v for k, v in blood.items()},
                **{f"brain_{k}": v for k, v in brain.items()},
            }
        )
        breakdown = _compute_task_breakdown(df)
        breakdown["task"] = task
        breakdown_rows.append(breakdown)
    if missing:
        print("Missing task files:", ", ".join(sorted(missing)))
    metrics_df = pd.DataFrame(metric_rows)
    breakdown_df = pd.DataFrame(breakdown_rows)
    return metrics_df, breakdown_df


def _load_task_datasets(task_files: Dict[str, Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for task, path in task_files.items():
        if not path.exists():
            missing.append(task)
            continue
        task_df = pd.read_csv(path)
        task_df = task_df.copy()
        task_df["task"] = task
        frames.append(task_df)
    if missing:
        print("Missing task comparison files:", ", ".join(sorted(missing)))
    if not frames:
        raise FileNotFoundError("No task comparison files available.")
    return pd.concat(frames, ignore_index=True)


def main(args: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare blood vs brain classifiers.")

    # Define default output directory
    default_out_dir = DEFAULT_RESULTS_DIR

    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL,
                        help="Which LOO model artifacts to load (rf/xgb/lr). Default: rf")
    parser.add_argument("--file-suffix", type=str, default="", help="Optional suffix for input/output CSV stems (e.g. _retrain)")
    parser.add_argument("--csv", type=Path, default=None, help="Optional combined comparison_results.csv")
    parser.add_argument("--out", type=Path, default=None, help="CSV to save metrics table (default: includes model name)")
    parser.add_argument("--counts-out", type=Path, default=None, help="CSV to save better-model counts (default: includes model name)")
    parser.add_argument("--tissue-out", type=Path, default=None, help="CSV to save tissue-specific blood/brain metrics (default: includes model name)")
    parser.add_argument("--tissue-model-task-out", type=Path, default=None, help="CSV to save all-models metrics for brain, blood_PCC, blood_DLPFC across AD_CONTROL and MCI_CONTROL")
    parser.add_argument("--tissue-model-task-group-out", type=Path, default=None, help="CSV to save grouped dataset-level metrics with group in {ALL,CONTROL,AD,MCI}")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR, help="Directory containing blood/brain log files")
    parser.add_argument("--log-raw-out", type=Path, default=None, help="CSV path for raw log metrics (default: includes model name)")
    parser.add_argument("--log-summary-out", type=Path, default=None, help="CSV path for log-based blood vs brain comparison (default: includes model name)")
    parser.add_argument("--task-metrics-out", type=Path, default=None, help="CSV path for per-task blood vs brain LOO metrics (default: includes model name)")
    parser.add_argument("--task-breakdown-out", type=Path, default=None, help="CSV path for per-task better-model breakdown (default: includes model name)")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH,
                        help="Path to metadata file containing individualID and cogdx")
    parser.add_argument("--cogdx-summary-out", type=Path, default=None, help="CSV path for cogdx failure summary (default: includes model name)")
    parser.add_argument("--cogdx-details-out", type=Path, default=None, help="CSV path for detailed cogdx failures (default: includes model name)")
    parsed = parser.parse_args(args=args)

    suffix = parsed.file_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"

    def with_suffix(path: Path) -> Path:
        if not suffix or path.stem.endswith(suffix):
            return path
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")

    # Build model-suffixed default output paths
    m = parsed.model
    # LOO-specific outputs (model suffix)
    out                 = with_suffix(parsed.out or default_out_dir / f"comparison_metrics_{m}.csv")
    counts_out          = with_suffix(parsed.counts_out or default_out_dir / f"comparison_better_counts_{m}.csv")
    tissue_out          = with_suffix(parsed.tissue_out or default_out_dir / f"comparison_metrics_by_tissue_{m}.csv")
    tissue_model_task_out = with_suffix(parsed.tissue_model_task_out or default_out_dir / "comparison_metrics_by_tissue_task_model.csv")
    tissue_model_task_group_out = with_suffix(parsed.tissue_model_task_group_out or default_out_dir / "comparison_metrics_by_tissue_task_model_group.csv")
    task_metrics_out    = with_suffix(parsed.task_metrics_out or default_out_dir / f"task_comparison_metrics_{m}.csv")
    task_breakdown_out  = with_suffix(parsed.task_breakdown_out or default_out_dir / f"task_comparison_breakdown_{m}.csv")
    cogdx_summary_out   = with_suffix(parsed.cogdx_summary_out or default_out_dir / f"cogdx_failure_summary_{m}.csv")
    cogdx_details_out   = with_suffix(parsed.cogdx_details_out or default_out_dir / f"cogdx_failure_details_{m}.csv")
    # Log-based outputs (all models, no model suffix)
    log_raw_out         = with_suffix(parsed.log_raw_out or default_out_dir / "log_model_metrics_raw.csv")
    log_summary_out     = with_suffix(parsed.log_summary_out or default_out_dir / "log_model_comparison.csv")

    task_files = build_task_result_files(parsed.model, file_suffix=suffix)
    summary, better_counts, combined_df = compare_models(parsed.csv, task_files=task_files)
    tissue_summary = build_tissue_metrics(combined_df)
    tissue_model_task_summary = build_task_model_tissue_metrics(file_suffix=suffix)
    tissue_model_task_group_summary = build_task_model_tissue_group_metrics(file_suffix=suffix)
    raw_logs, log_comparison = build_log_comparison(parsed.log_dir)
    print(f"\nUsing model: {parsed.model}")
    for key, path in task_files.items():
        print(f"  {key}: {path} {'[found]' if path.exists() else '[MISSING]'}")
    task_metrics, task_breakdown = compare_tasks(task_files)
    cogdx_summary = pd.DataFrame()
    cogdx_details = pd.DataFrame()
    if parsed.metadata and parsed.metadata.exists():
        try:
            metadata = load_metadata(parsed.metadata)
            cogdx_summary, cogdx_details = analyze_cogdx_failures(combined_df, metadata)
        except Exception as exc:
            print(f"Unable to analyze cogdx failures: {exc}")
    else:
        print(f"Metadata file {parsed.metadata} not found; skipping cogdx analysis.")

    if log_comparison.empty:
        print("No log files found for log-based comparison.")
    else:
        print("\n=== Log-Based Model Comparison ===")
        print(log_comparison.to_string(index=False))

    if not task_metrics.empty:
        print("\n=== Task-Level LOO Metrics ===")
        print(task_metrics.to_string(index=False))
    if not tissue_summary.empty:
        print("\n=== Tissue-Specific Model Metrics ===")
        print(tissue_summary.to_string(float_format="%.4f"))
    elif "task" not in combined_df.columns:
        print("\nNo task column found in combined data; skipping tissue-specific metrics.")
    if not task_breakdown.empty:
        print("\n=== Task-Level Better-Model Counts ===")
        print(task_breakdown.to_string(index=False))
    if not tissue_model_task_summary.empty:
        print("\n=== Dataset-Level Blood/Brain Metrics by Task and Model ===")
        print(tissue_model_task_summary.to_string(index=False, float_format="%.4f"))
    if not tissue_model_task_group_summary.empty:
        print("\n=== Dataset-Level Grouped Blood/Brain Metrics ===")
        print(tissue_model_task_group_summary.to_string(index=False, float_format="%.4f"))
    if not cogdx_summary.empty:
        print("\n=== cogdx Distribution for Dual Failures ===")
        print(cogdx_summary.to_string(index=False, float_format="%.4f"))

    # Always save outputs to files
    export_results(summary, out)
    if not tissue_summary.empty:
        export_results(tissue_summary, tissue_out)
    if not tissue_model_task_summary.empty:
        tissue_model_task_summary.to_csv(tissue_model_task_out, index=False, float_format="%.4f")
        print(f"Saved 3-tissue task/model metrics to {tissue_model_task_out}")
    if not tissue_model_task_group_summary.empty:
        tissue_model_task_group_summary.to_csv(tissue_model_task_group_out, index=False, float_format="%.4f")
        print(f"Saved grouped task/model metrics to {tissue_model_task_group_out}")
    export_counts(better_counts, counts_out)
    export_log_tables(raw_logs, log_comparison, log_raw_out, log_summary_out)
    if not task_metrics.empty:
        task_metrics.to_csv(task_metrics_out, index=False, float_format="%.4f")
        print(f"Saved task-level metrics to {task_metrics_out}")
    if not task_breakdown.empty:
        task_breakdown.to_csv(task_breakdown_out, index=False)
        print(f"Saved task-level breakdown to {task_breakdown_out}")
    if not cogdx_summary.empty:
        cogdx_summary.to_csv(cogdx_summary_out, index=False, float_format="%.4f")
        print(f"Saved cogdx failure summary to {cogdx_summary_out}")
    if not cogdx_details.empty:
        cogdx_details.to_csv(cogdx_details_out, index=False)
        print(f"Saved cogdx failure details to {cogdx_details_out}")


if __name__ == "__main__":
    main()
