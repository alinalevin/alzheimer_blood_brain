"""Utility helpers for comparing blood and brain classifiers on shared samples."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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

TASK_RESULT_FILES = {
    "AD_CONTROL_DLPFC": LOG_DIR / "comparison_results_AD_CONTROL_DLPFC.csv",
    "AD_CONTROL_PCC":   LOG_DIR / "comparison_results_AD_CONTROL_PCC.csv",
    "MCI_CONTROL_DLPFC": LOG_DIR / "comparison_results_MCI_CONTROL_DLPFC.csv",
    "MCI_CONTROL_PCC":   LOG_DIR / "comparison_results_MCI_CONTROL_PCC.csv",
}

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


def _find_latest_log(log_dir: Path, prefix: str) -> Optional[Path]:
    candidates = sorted(log_dir.glob(f"{prefix}_*.log"))
    if candidates:
        latest = candidates[-1]
        print(f"Using log file: {latest}")
        return latest
    else:
        print(f"No log file found matching pattern: {prefix}_*.log")
        return None


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
            log_path = _find_latest_log(log_dir, prefix)
            if not log_path:
                continue
            df = _parse_log_file(log_path, tissue, task)
            if not df.empty:
                frames.append(df)
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


def compare_models(csv_path: Optional[Path] = None) -> tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    if csv_path is not None:
        try:
            df = _load_data(csv_path)
        except FileNotFoundError:
            print(f"Overall comparison file {csv_path} not found, falling back to task-level CSVs.")
            df = _load_task_datasets(TASK_RESULT_FILES)
    else:
        df = _load_task_datasets(TASK_RESULT_FILES)
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

    parser.add_argument("--csv", type=Path, default=None, help="Optional combined comparison_results.csv")
    parser.add_argument("--out", type=Path, default=default_out_dir / "comparison_metrics.csv", help="CSV to save metrics table")
    parser.add_argument("--counts-out", type=Path, default=default_out_dir / "comparison_better_counts.csv",
                        help="CSV to save better-model counts")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR, help="Directory containing blood/brain log files")
    parser.add_argument("--log-raw-out", type=Path, default=default_out_dir / "log_model_metrics_raw.csv",
                        help="CSV path for raw log metrics")
    parser.add_argument("--log-summary-out", type=Path, default=default_out_dir / "log_model_comparison.csv",
                        help="CSV path for log-based blood vs brain comparison")
    parser.add_argument("--task-metrics-out", type=Path, default=default_out_dir / "task_comparison_metrics.csv",
                        help="CSV path for per-task blood vs brain LOO metrics")
    parser.add_argument("--task-breakdown-out", type=Path, default=default_out_dir / "task_comparison_breakdown.csv",
                        help="CSV path for per-task better-model breakdown")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH,
                        help="Path to metadata file containing individualID and cogdx")
    parser.add_argument("--cogdx-summary-out", type=Path, default=default_out_dir / "cogdx_failure_summary.csv",
                        help="CSV path for cogdx failure summary")
    parser.add_argument("--cogdx-details-out", type=Path, default=default_out_dir / "cogdx_failure_details.csv",
                        help="CSV path for detailed cogdx failures")
    parsed = parser.parse_args(args=args)

    summary, better_counts, combined_df = compare_models(parsed.csv)
    raw_logs, log_comparison = build_log_comparison(parsed.log_dir)
    task_metrics, task_breakdown = compare_tasks(TASK_RESULT_FILES)
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
    if not task_breakdown.empty:
        print("\n=== Task-Level Better-Model Counts ===")
        print(task_breakdown.to_string(index=False))
    if not cogdx_summary.empty:
        print("\n=== cogdx Distribution for Dual Failures ===")
        print(cogdx_summary.to_string(index=False, float_format="%.4f"))

    # Always save outputs to files
    export_results(summary, parsed.out)
    export_counts(better_counts, parsed.counts_out)
    export_log_tables(raw_logs, log_comparison, parsed.log_raw_out, parsed.log_summary_out)
    if not task_metrics.empty:
        task_metrics.to_csv(parsed.task_metrics_out, index=False, float_format="%.4f")
        print(f"Saved task-level metrics to {parsed.task_metrics_out}")
    if not task_breakdown.empty:
        task_breakdown.to_csv(parsed.task_breakdown_out, index=False)
        print(f"Saved task-level breakdown to {parsed.task_breakdown_out}")
    if not cogdx_summary.empty:
        cogdx_summary.to_csv(parsed.cogdx_summary_out, index=False, float_format="%.4f")
        print(f"Saved cogdx failure summary to {parsed.cogdx_summary_out}")
    if not cogdx_details.empty:
        cogdx_details.to_csv(parsed.cogdx_details_out, index=False)
        print(f"Saved cogdx failure details to {parsed.cogdx_details_out}")


if __name__ == "__main__":
    main()
