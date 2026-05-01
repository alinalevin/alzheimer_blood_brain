"""Compare blood and brain classifiers on shared test subjects.

This script loads the persisted test splits and fitted models produced by
`blood_classifier.py` and `brain_classifier.py`, re-computes predictions, and
reports which tissue-specific classifier performed better for each subject.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.utils.class_weight import compute_sample_weight

# Suppress noisy FutureWarnings from sklearn/imbalanced-learn version mismatches
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BLOOD_DIR = BASE_DIR / "files" / "blood"
DEFAULT_BRAIN_DIR = BASE_DIR / "files" / "brain"
DEFAULT_OUTPUT_PATH = BASE_DIR / "files" / "comparison_results.csv"
BLOOD_METADATA_FILE = "blood_final_residuals_metadata_all.xlsx"
BRAIN_METADATA_FILE = "brain_metadata_after_preprocess_all.xlsx"

TASK_LABELS: Dict[str, Dict[str, int]] = {
    "AD_CONTROL": {"CONTROL": 0, "AD": 1},
    "MCI_CONTROL": {"CONTROL": 0, "MCI": 1},
}

MODEL_CHOICES = ("rf", "xgb", "lr")


@dataclass
class TissueModelBundle:
    name: str
    model: object
    features: pd.DataFrame
    labels: pd.Series
    task: str

    @property
    def specimen_ids(self) -> pd.Index:
        return self.features.index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-out comparison between blood and brain classifiers.")
    parser.add_argument("--blood-dir", type=Path, default=DEFAULT_BLOOD_DIR, help="Directory containing blood classifier artifacts")
    parser.add_argument("--brain-dir", type=Path, default=DEFAULT_BRAIN_DIR, help="Directory containing brain classifier artifacts")
    parser.add_argument("--task", choices=list(TASK_LABELS.keys()), default="AD_CONTROL", help="Phenotype task to compare")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="rf", help="Base model to load (rf/xgb/lr)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to store the per-patient comparison table")
    parser.add_argument(
        "--preview-output",
        type=Path,
        help="Optional path to save the raw merged blood/brain predictions before comparison",
    )
    return parser.parse_args()


def build_prefix(tissue: str, model: str) -> str:
    if tissue == "blood":
        return f"{model}_blood"
    return f"brain_{model}"


def load_bundle(base_dir: Path, prefix: str, task: str) -> TissueModelBundle:
    model_path = base_dir / f"{prefix}_{task}.pkl"
    x_train_path = base_dir / f"x_train_{prefix}_{task}.csv"
    y_train_path = base_dir / f"y_train_{prefix}_{task}.csv"
    x_test_path = base_dir / f"x_test_{prefix}_{task}.csv"
    y_test_path = base_dir / f"y_test_{prefix}_{task}.csv"

    required_files = [model_path, x_train_path, y_train_path, x_test_path, y_test_path]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")

    model = joblib.load(model_path)
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

    return TissueModelBundle(prefix, model, features, labels, task)


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


def leave_one_out_predict(bundle: TissueModelBundle) -> pd.DataFrame:
    X = bundle.features
    y = bundle.labels.loc[X.index].astype(int)
    n = len(X)
    indices = np.arange(n)
    predictions = np.zeros(n, dtype=int)
    probabilities = np.full(n, np.nan, dtype=float)
    loo = LeaveOneOut()

    print(f"  Running LOO on {n} samples for {bundle.name}...", flush=True)
    for i, (train_idx, test_idx) in enumerate(loo.split(indices)):
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"    Progress: {i + 1}/{n}", flush=True)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        est = clone(bundle.model)
        sample_weight = compute_sample_weight("balanced", y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                est.fit(X_train, y_train, sample_weight=sample_weight)
            except TypeError:
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


def describe_intersection(blood_bundle: TissueModelBundle, brain_bundle: TissueModelBundle, brain_metadata: Optional[pd.DataFrame] = None) -> None:
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
    if brain_metadata is not None and 'tissue' in brain_metadata.columns:
        shared_ids = brain_bundle.features.index
        brain_meta_subset = brain_metadata[brain_metadata['individual_clean'].isin(shared_ids)]
        if not brain_meta_subset.empty:
            region_counts = brain_meta_subset['tissue'].value_counts()
            print("[Brain Regions] in intersection:")
            for region, count in region_counts.items():
                print(f"  {region}: {count} samples")
        else:
            print("[Brain Regions] Could not map samples to metadata for region breakdown")


def main() -> None:
    args = parse_args()
    blood_prefix = build_prefix("blood", args.model)
    brain_prefix = build_prefix("brain", args.model)

    blood_bundle = load_bundle(args.blood_dir, blood_prefix, args.task)
    brain_bundle = load_bundle(args.brain_dir, brain_prefix, args.task)

    blood_metadata = load_metadata_frame(args.blood_dir / BLOOD_METADATA_FILE)
    brain_metadata = load_metadata_frame(args.brain_dir / BRAIN_METADATA_FILE)
    blood_bundle = align_bundle_to_individuals(blood_bundle, blood_metadata)
    brain_bundle = align_bundle_to_individuals(brain_bundle, brain_metadata)

    # Identify unique tissues in brain metadata
    if 'tissue' not in brain_metadata.columns:
        raise ValueError("'tissue' column missing from brain metadata. Cannot split by DLPFC/PCC.")
    tissues = ["DLPFC", "PCC"]
    for tissue in tissues:
        print(f"\n=== Running Blood vs {tissue} comparison ===")
        # Filter brain metadata and bundle for this tissue
        brain_meta_tissue = brain_metadata[brain_metadata['tissue'].str.upper() == tissue.upper()]
        if brain_meta_tissue.empty:
            print(f"No samples found for tissue {tissue} in brain metadata. Skipping.")
            continue
        brain_ids_tissue = pd.Index(brain_meta_tissue['individual_clean'].unique())
        # Subset brain bundle to only these individuals
        brain_bundle_tissue = subset_bundle(brain_bundle, brain_ids_tissue.intersection(brain_bundle.features.index))
        # Find intersection with blood
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
            out_path = args.preview_output.parent / f"preview_merged_{args.task}_{tissue}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged_preview.to_csv(out_path)
            print(f"Preview merged predictions saved to {out_path}")

        # Save comparison results with tissue in filename
        output_path = args.output.parent / f"comparison_results_{args.task}_{tissue}.csv"
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


if __name__ == "__main__":
    main()
