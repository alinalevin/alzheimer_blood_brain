"""Compare blood and brain classifiers on shared subjects using fold-wise retraining.

This script loads persisted train/test splits produced by `blood_classifier.py` and
`brain_classifier.py`, rebuilds model instances with the same tuned settings, and
runs leave-one-out (LOO) predictions to avoid leakage from previously fit `.pkl`
models.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Suppress noisy FutureWarnings from sklearn/imbalanced-learn version mismatches
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BLOOD_DIR = BASE_DIR / "files" / "blood"
DEFAULT_BRAIN_DIR = BASE_DIR / "files" / "brain"
DEFAULT_OUTPUT_PATH = BASE_DIR / "files" / "comparison_results_retrain.csv"
DEFAULT_LOG_PATH = BASE_DIR / "files" / "leave_one_out_retrain.log"
BLOOD_METADATA_FILE = "blood_final_residuals_metadata_all.xlsx"
BRAIN_METADATA_FILE = "brain_metadata_after_preprocess_all.xlsx"

TASK_LABELS: Dict[str, Dict[str, int]] = {
    "AD_CONTROL": {"CONTROL": 0, "AD": 1},
    "MCI_CONTROL": {"CONTROL": 0, "MCI": 1},
}

MODEL_CHOICES = ("rf", "xgb", "lr")


class Tee:
    """Mirror stdout/stderr to both terminal and a log file."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class TissueModelBundle:
    name: str
    tissue: str
    model_key: str
    features: pd.DataFrame
    labels: pd.Series
    task: str

    @property
    def specimen_ids(self) -> pd.Index:
        return self.features.index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-out comparison between blood and brain classifiers with fold-wise retraining.")
    parser.add_argument("--blood-dir", type=Path, default=DEFAULT_BLOOD_DIR, help="Directory containing blood classifier artifacts")
    parser.add_argument("--brain-dir", type=Path, default=DEFAULT_BRAIN_DIR, help="Directory containing brain classifier artifacts")
    parser.add_argument("--task", choices=list(TASK_LABELS.keys()), default="AD_CONTROL", help="Phenotype task to compare")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="rf", help="Model family to retrain inside each LOO fold (rf/xgb/lr)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Base path used for per-patient comparison tables")
    parser.add_argument(
        "--preview-output",
        type=Path,
        help="Optional base path to save merged blood/brain prediction tables",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to run log file. Filename should include _retrain.",
    )
    return parser.parse_args()


def ensure_retrain_in_filename(path: Path) -> Path:
    if "_retrain" in path.stem:
        return path
    return path.with_name(f"{path.stem}_retrain{path.suffix}")


def build_prefix(tissue: str, model: str) -> str:
    if tissue == "blood":
        return f"{model}_blood"
    return f"brain_{model}"


def load_bundle(base_dir: Path, prefix: str, task: str, tissue: str, model_key: str) -> TissueModelBundle:
    x_train_path = base_dir / f"x_train_{prefix}_{task}.csv"
    y_train_path = base_dir / f"y_train_{prefix}_{task}.csv"
    x_test_path = base_dir / f"x_test_{prefix}_{task}.csv"
    y_test_path = base_dir / f"y_test_{prefix}_{task}.csv"

    required_files = [x_train_path, y_train_path, x_test_path, y_test_path]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")

    x_train = pd.read_csv(x_train_path, index_col=0)
    x_test = pd.read_csv(x_test_path, index_col=0)
    y_train = pd.read_csv(y_train_path, index_col=0)["condition"].astype(int)
    y_test = pd.read_csv(y_test_path, index_col=0)["condition"].astype(int)

    features = pd.concat([x_train, x_test], axis=0)
    labels = pd.concat([y_train, y_test], axis=0)
    # Remove duplicates while preserving the last occurrence (test set should override train if overlaps)
    features = features[~features.index.duplicated(keep="last")]
    labels = labels[~labels.index.duplicated(keep="last")]
    labels = labels.reindex(features.index)

    return TissueModelBundle(prefix, tissue, model_key, features, labels, task)


def load_metadata_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    metadata = pd.read_excel(path).copy()
    if "specimenID" not in metadata.columns:
        raise ValueError(f"specimenID column missing from metadata: {path}")
    metadata["specimenID"] = metadata["specimenID"].astype(str)
    if "individualID" in metadata.columns:
        metadata["individual_clean"] = metadata["individualID"].fillna(metadata["specimenID"]).astype(str)
    else:
        metadata["individual_clean"] = metadata["specimenID"]
    return metadata


def align_bundle_to_individuals(bundle: TissueModelBundle, metadata: pd.DataFrame) -> TissueModelBundle:
    rename_map = metadata.set_index("specimenID")["individual_clean"].to_dict()
    features = bundle.features.rename(index=rename_map)
    labels = bundle.labels.rename(index=rename_map)
    # Drop duplicate individuals keeping the last (test samples override train)
    duplicated = features.index.duplicated(keep="last")
    if duplicated.any():
        features = features.loc[~duplicated]
        labels = labels.loc[features.index]
    return replace(bundle, features=features, labels=labels)


def decode_labels(series: pd.Series, task: str) -> pd.Series:
    inverse = {v: k for k, v in TASK_LABELS[task].items()}
    return series.map(inverse)


def compute_scale_pos_weight(y: pd.Series) -> float:
    values, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(values, counts))
    negatives = class_counts.get(0, 0)
    positives = class_counts.get(1, 1)
    return float(negatives) / float(positives) if positives else 1.0


def build_estimator(tissue: str, model_key: str, task: str, y_train: pd.Series):
    if model_key == "rf":
        if tissue == "blood":
            params = {
                "n_estimators": 300 if task == "AD_CONTROL" else 150,
                "criterion": "entropy",
                "random_state": 42,
                "sampling_strategy": "not minority" if task == "AD_CONTROL" else "all",
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
                "bootstrap": False,
                "n_jobs": 8,
            }
            if task == "AD_CONTROL":
                params["min_samples_split"] = 5
        else:
            params = {
                "n_estimators": 200,
                "criterion": "entropy",
                "random_state": 42,
                "sampling_strategy": "all" if task == "MCI_CONTROL" else "not minority",
                "max_features": "sqrt",
                "class_weight": "balanced",
                "bootstrap": task == "AD_CONTROL",
                "n_jobs": 8,
                "min_samples_split": 2,
            }
            if task == "AD_CONTROL":
                params["min_samples_leaf"] = 1
            else:
                params["min_samples_leaf"] = 2
                params["max_depth"] = 9
        return BalancedRandomForestClassifier(**params)

    if model_key == "xgb":
        if tissue == "blood":
            params = {
                "n_estimators": 150,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": 42,
                "learning_rate": 0.2 if task == "AD_CONTROL" else 0.25,
                "max_depth": 4 if task == "AD_CONTROL" else 3,
                "min_child_weight": 1,
                "colsample_bytree": 0.1 if task == "AD_CONTROL" else 0.2,
                "n_jobs": 8,
                "scale_pos_weight": compute_scale_pos_weight(y_train),
            }
        else:
            params = {
                "n_estimators": 200,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": 42,
                "learning_rate": 0.2,
                "max_depth": 4,
                "min_child_weight": 1,
                "colsample_bytree": 0.2,
                "n_jobs": 8,
                "scale_pos_weight": compute_scale_pos_weight(y_train),
            }
        return xgb.XGBClassifier(**params)

    max_iter = 300 if tissue == "blood" else 400
    lr = LogisticRegression(
        random_state=42,
        fit_intercept=True,
        class_weight="balanced",
        max_iter=max_iter,
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.1,
        C=0.1,
        n_jobs=8,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", lr)])


def leave_one_out_predict(bundle: TissueModelBundle) -> pd.DataFrame:
    X = bundle.features
    y = bundle.labels.loc[X.index].astype(int)
    n = len(X)
    indices = np.arange(n)
    predictions = np.zeros(n, dtype=int)
    probabilities = np.full(n, np.nan, dtype=float)
    loo = LeaveOneOut()

    print(f"  Running retrained LOO on {n} samples for {bundle.name}...", flush=True)
    for i, (train_idx, test_idx) in enumerate(loo.split(indices)):
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"    Progress: {i + 1}/{n}", flush=True)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        est = build_estimator(bundle.tissue, bundle.model_key, bundle.task, y_train)
        sample_weight = compute_sample_weight("balanced", y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                est.fit(X_train, y_train, sample_weight=sample_weight)
            except (TypeError, ValueError):
                # sklearn Pipeline expects step-scoped sample_weight (e.g. clf__sample_weight)
                if isinstance(est, Pipeline):
                    try:
                        est.fit(X_train, y_train, clf__sample_weight=sample_weight)
                    except TypeError:
                        est.fit(X_train, y_train)
                else:
                    est.fit(X_train, y_train)
        pred = est.predict(X_test)[0]
        predictions[test_idx[0]] = pred
        if hasattr(est, "predict_proba"):
            probabilities[test_idx[0]] = est.predict_proba(X_test)[0, 1]

    result = pd.DataFrame(
        {
            "true_label": decode_labels(y, bundle.task).to_numpy(),
            "prediction": predictions,
            "probability": probabilities,
        },
        index=X.index,
    )
    return result


def merge_prediction_frames(blood_df: pd.DataFrame, brain_df: pd.DataFrame) -> pd.DataFrame:
    merged = blood_df.join(
        brain_df,
        how="inner",
        lsuffix="_blood",
        rsuffix="_brain",
    )
    if merged.empty:
        raise ValueError("No overlapping specimen IDs between blood and brain test sets")
    return merged


def encode_labels(series: pd.Series, task: str) -> pd.Series:
    mapping = TASK_LABELS[task]
    if set(series.dropna().unique()).issubset(set(mapping.values())):
        return series.astype(int)
    try:
        return series.map(mapping).astype(int)
    except KeyError as exc:
        missing = set(series.unique()) - set(mapping.keys())
        raise ValueError(f"Unexpected labels {missing} for task {task}") from exc


def compare_predictions(
    merged: pd.DataFrame,
    task: str,
) -> pd.DataFrame:
    merged = merged.copy()
    merged["true_label_num"] = encode_labels(merged["true_label_blood"], task)
    merged["blood_correct"] = merged["prediction_blood"].astype(int) == merged["true_label_num"]
    merged["brain_correct"] = merged["prediction_brain"].astype(int) == merged["true_label_num"]

    def winner(row: pd.Series) -> str:
        if row["blood_correct"] and not row["brain_correct"]:
            return "blood"
        if row["brain_correct"] and not row["blood_correct"]:
            return "brain"
        if row["blood_correct"] and row["brain_correct"]:
            return "both"
        return "neither"

    merged["better_model"] = merged.apply(winner, axis=1)
    return merged


def subset_bundle(bundle: TissueModelBundle, specimen_ids: pd.Index) -> TissueModelBundle:
    subset_features = bundle.features.loc[specimen_ids]
    subset_features = subset_features[~subset_features.index.duplicated(keep="first")]
    subset_labels = bundle.labels.loc[subset_features.index]
    return replace(bundle, features=subset_features, labels=subset_labels)


def describe_intersection(
    blood_bundle: TissueModelBundle,
    brain_bundle: TissueModelBundle,
    brain_metadata: Optional[pd.DataFrame] = None,
) -> None:
    """Print shared sample count and gene dimensions for both tissues."""
    shared_samples = len(blood_bundle.features)
    blood_genes = blood_bundle.features.shape[1]
    brain_genes = brain_bundle.features.shape[1]
    print(
        "[Intersection] Shared samples: {shared} | Blood genes: {blood_genes} | Brain genes: {brain_genes}".format(
            shared=shared_samples,
            blood_genes=blood_genes,
            brain_genes=brain_genes,
        )
    )

    # Report brain region breakdown if metadata is available
    if brain_metadata is not None and "tissue" in brain_metadata.columns:
        shared_ids = brain_bundle.features.index
        brain_meta_subset = brain_metadata[brain_metadata["individual_clean"].isin(shared_ids)]
        if not brain_meta_subset.empty:
            region_counts = brain_meta_subset["tissue"].value_counts()
            print("[Brain Regions] in intersection:")
            for region, count in region_counts.items():
                print(f"  {region}: {count} samples")
        else:
            print("[Brain Regions] Could not map samples to metadata for region breakdown")


def run(args: argparse.Namespace) -> None:
    blood_prefix = build_prefix("blood", args.model)
    brain_prefix = build_prefix("brain", args.model)

    blood_bundle = load_bundle(args.blood_dir, blood_prefix, args.task, tissue="blood", model_key=args.model)
    brain_bundle = load_bundle(args.brain_dir, brain_prefix, args.task, tissue="brain", model_key=args.model)

    blood_metadata = load_metadata_frame(args.blood_dir / BLOOD_METADATA_FILE)
    brain_metadata = load_metadata_frame(args.brain_dir / BRAIN_METADATA_FILE)
    blood_bundle = align_bundle_to_individuals(blood_bundle, blood_metadata)
    brain_bundle = align_bundle_to_individuals(brain_bundle, brain_metadata)

    if "tissue" not in brain_metadata.columns:
        raise ValueError("'tissue' column missing from brain metadata. Cannot split by DLPFC/PCC.")

    tissues = ["DLPFC", "PCC"]
    for tissue in tissues:
        print(f"\n=== Running Blood vs {tissue} comparison (retrain) ===")
        brain_meta_tissue = brain_metadata[brain_metadata["tissue"].str.upper() == tissue.upper()]
        if brain_meta_tissue.empty:
            print(f"No samples found for tissue {tissue} in brain metadata. Skipping.")
            continue

        brain_ids_tissue = pd.Index(brain_meta_tissue["individual_clean"].unique())
        brain_bundle_tissue = subset_bundle(brain_bundle, brain_ids_tissue.intersection(brain_bundle.features.index))

        shared_ids = blood_bundle.specimen_ids[blood_bundle.specimen_ids.isin(brain_bundle_tissue.specimen_ids)]
        shared_ids = pd.Index(shared_ids.unique())
        if shared_ids.empty:
            print(f"No overlapping specimen IDs between blood and {tissue} datasets before prediction. Skipping.")
            continue

        print(f"Found {len(shared_ids)} shared specimens for Blood vs {tissue}.")
        blood_bundle_tissue = subset_bundle(blood_bundle, shared_ids)
        brain_bundle_tissue = subset_bundle(brain_bundle_tissue, shared_ids)
        describe_intersection(blood_bundle_tissue, brain_bundle_tissue, brain_metadata)

        blood_eval = leave_one_out_predict(blood_bundle_tissue)
        brain_eval = leave_one_out_predict(brain_bundle_tissue)

        merged_preview = merge_prediction_frames(blood_eval, brain_eval)
        if args.preview_output:
            out_path = args.preview_output.parent / f"preview_merged_{args.task}_{tissue}_{args.model}_retrain.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged_preview.to_csv(out_path)
            print(f"Preview merged predictions saved to {out_path}")

        output_path = args.output.parent / f"comparison_results_{args.task}_{tissue}_{args.model}_retrain.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison = compare_predictions(merged_preview, args.task)
        comparison.to_csv(output_path)

        summary = comparison["better_model"].value_counts().to_dict()
        total = len(comparison)
        print(f"Compared {total} shared specimens for Blood vs {tissue}. Distribution of better models:")
        for key in ("blood", "brain", "both", "neither"):
            count = summary.get(key, 0)
            print(f"  {key:<7}: {count:>3} ({count / total:.1%})")
        print(f"Detailed results saved to {output_path}")


def main() -> None:
    args = parse_args()
    args.log_output = ensure_retrain_in_filename(args.log_output)
    args.log_output.parent.mkdir(parents=True, exist_ok=True)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    with args.log_output.open("w", encoding="utf-8") as log_file:
        tee = Tee(orig_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            print(f"Writing retrain run log to {args.log_output}")
            run(args)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


if __name__ == "__main__":
    main()

