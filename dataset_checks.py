"""Summarize blood, brain, and shared datasets for classifier inputs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
BLOOD_DIR_DEFAULT = BASE_DIR / "files" / "blood"
BRAIN_DIR_DEFAULT = BASE_DIR / "files" / "brain"

BLOOD_COUNTS_FILE = "blood_r1_clean_all.csv"
BLOOD_METADATA_FILE = "blood_final_residuals_metadata_all.xlsx"
BRAIN_COUNTS_FILE = "brain_residuals_all.csv"
BRAIN_METADATA_FILE = "brain_metadata_after_preprocess_all.xlsx"

# Per-tissue DGE gene lists (ensembl_gene_id column)
DLPFC_AD_GENES_FILE = "DLPFC_dge_ad_control_meta_results.xlsx"
DLPFC_MCI_GENES_FILE = "DLPFC_dge_meta_results_MCI_AD.xlsx"
PCC_AD_GENES_FILE = "PCC_dge_meta_results_AD_CONTROL.xlsx"
PCC_MCI_GENES_FILE = "PCC_dge_meta_results_MCI_AD.xlsx"

# Map tissue name -> {task -> gene file basename}
TISSUE_GENE_FILES: dict[str, dict[str, str]] = {
    "DLPFC": {"AD_CONTROL": DLPFC_AD_GENES_FILE, "MCI_CONTROL": DLPFC_MCI_GENES_FILE},
    "PCC":   {"AD_CONTROL": PCC_AD_GENES_FILE,   "MCI_CONTROL": PCC_MCI_GENES_FILE},
}


@dataclass
class DatasetPaths:
    counts: Path
    metadata: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset diagnostics for blood/brain pipelines")
    parser.add_argument("--blood-dir", type=Path, default=BLOOD_DIR_DEFAULT, help="Directory containing blood data files")
    parser.add_argument("--brain-dir", type=Path, default=BRAIN_DIR_DEFAULT, help="Directory containing brain data files")
    parser.add_argument("--blood-counts", type=Path, default=None, help="Override path to blood counts CSV")
    parser.add_argument("--blood-metadata", type=Path, default=None, help="Override path to blood metadata XLSX")
    parser.add_argument("--brain-counts", type=Path, default=None, help="Override path to brain counts CSV")
    parser.add_argument("--brain-metadata", type=Path, default=None, help="Override path to brain metadata XLSX")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[DatasetPaths, DatasetPaths]:
    blood_counts = args.blood_counts or args.blood_dir / BLOOD_COUNTS_FILE
    blood_metadata = args.blood_metadata or args.blood_dir / BLOOD_METADATA_FILE
    brain_counts = args.brain_counts or args.brain_dir / BRAIN_COUNTS_FILE
    brain_metadata = args.brain_metadata or args.brain_dir / BRAIN_METADATA_FILE
    return DatasetPaths(blood_counts, blood_metadata), DatasetPaths(brain_counts, brain_metadata)


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    df = pd.read_excel(path)
    if "specimenID" not in df.columns:
        raise ValueError(f"specimenID column missing in {path}")
    df = df.dropna(subset=["specimenID"]).copy()
    df["specimenID"] = df["specimenID"].astype(str)
    if "individualID" in df.columns:
        df["individual_clean"] = df["individualID"].astype(str)
        mask = df["individualID"].isna()
        if mask.any():
            df.loc[mask, "individual_clean"] = df.loc[mask, "specimenID"]
    else:
        df["individual_clean"] = df["specimenID"]
    return df


def load_counts(path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing counts file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Counts file is empty: {path}")
    first_col = df.columns[0]
    if first_col != "ensembl_gene_id":
        df = df.rename(columns={first_col: "ensembl_gene_id"})
    df = df.set_index("ensembl_gene_id", drop=True)
    specimen_ids = metadata["specimenID"].tolist()
    available = [col for col in specimen_ids if col in df.columns]
    if not available:
        raise ValueError("No overlapping specimen IDs between metadata and counts")
    df = df[available]
    rename_map = metadata.set_index("specimenID")["individual_clean"].to_dict()
    df = df.rename(columns=rename_map)
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).first().T
    return df


def load_dge_gene_sets(brain_dir: Path) -> dict[str, dict[str, set[str]]]:
    """Load per-tissue DGE gene sets.

    Returns a nested dict:  tissue -> task -> set of base ensembl IDs (no version suffix).
    """
    result: dict[str, dict[str, set[str]]] = {}
    for tissue, task_files in TISSUE_GENE_FILES.items():
        result[tissue] = {}
        for task, fname in task_files.items():
            path = brain_dir / fname
            if not path.exists():
                print(f"  [warn] DGE gene file not found: {path}")
                continue
            df = pd.read_excel(path)
            if "ensembl_gene_id" not in df.columns:
                print(f"  [warn] ensembl_gene_id column missing in {path}")
                continue
            # Strip version suffix (e.g. ENSG00000000003.14 -> ENSG00000000003)
            ids = df["ensembl_gene_id"].dropna().astype(str).str.split(".").str[0]
            result[tissue][task] = set(ids)
    return result


def detect_condition_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("condition", "diagnosis", "dx", "DX", "phenotype"):
        if candidate in df.columns:
            return candidate
    return None


def summarize_conditions(label: str, metadata: pd.DataFrame) -> None:
    column = detect_condition_column(metadata)
    if not column:
        print(f"         {label} condition counts: not available (column missing)")
        return
    counts = metadata[column].value_counts(dropna=False).sort_index()
    print(f"         {label} condition counts:")
    for cond, cnt in counts.items():
        name = "Unknown" if pd.isna(cond) else str(cond)
        print(f"           - {name}: {cnt}")


def detect_age_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("AGE", "age", "Age", "AGE_AT_DEATH", "AGE_AT_VISIT"):
        if candidate in df.columns:
            return candidate
    return None


def summarize_age(label: str, metadata: pd.DataFrame) -> None:
    column = detect_age_column(metadata)
    if not column:
        print(f"         {label} age stats: not available (column missing)")
        return
    series = pd.to_numeric(metadata[column], errors="coerce").dropna()
    if series.empty:
        print(f"         {label} age stats: no numeric values")
        return
    desc = series.describe(percentiles=[0.25, 0.5, 0.75])
    print(
        f"         {label} age range: min={desc['min']:.1f}, Q1={desc['25%']:.1f}, median={desc['50%']:.1f}, "
        f"Q3={desc['75%']:.1f}, max={desc['max']:.1f}"
    )


def summarize_dataset(label: str, counts: pd.DataFrame, metadata: pd.DataFrame) -> None:
    genes = counts.shape[0]
    samples = counts.shape[1]
    specimens = metadata["specimenID"].nunique()
    individuals = metadata["individual_clean"].nunique()
    print(f"[{label}] Genes: {genes:,} | Count matrix samples: {samples:,}")
    print(f"         Metadata specimens: {specimens:,} | Individuals: {individuals:,}")
    summarize_conditions(label, metadata)
    summarize_age(label, metadata)


def summarize_dataset_by_tissue(
    label: str,
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    tissue_col: str = "tissue",
    dge_gene_sets: Optional[dict[str, dict[str, set[str]]]] = None,
) -> None:
    """Call summarize_dataset once per tissue type found in metadata.

    If dge_gene_sets is provided (tissue -> task -> set of gene IDs), also
    reports how many DGE-selected genes are present in the counts matrix per task.
    """
    if tissue_col not in metadata.columns:
        print(f"[{label}] tissue column '{tissue_col}' not found – running combined summary.")
        summarize_dataset(label, counts, metadata)
        return
    tissues = sorted(metadata[tissue_col].dropna().unique())
    # Strip version suffix from counts index once for matching
    counts_ids_base = counts.index.str.split(".").str[0]
    for tissue in tissues:
        tissue_meta = metadata[metadata[tissue_col] == tissue].copy()
        tissue_individuals = tissue_meta["individual_clean"].tolist()
        available_cols = [c for c in tissue_individuals if c in counts.columns]
        if not available_cols:
            print(f"[{label} – {tissue}] No matching columns in counts matrix.")
            continue
        tissue_counts = counts[available_cols]
        # Filter rows to DGE genes for this tissue (union across tasks)
        if dge_gene_sets and tissue in dge_gene_sets:
            all_tissue_genes: set[str] = set()
            for gene_set in dge_gene_sets[tissue].values():
                all_tissue_genes |= gene_set
            # Match against base IDs (strip version suffix)
            mask = counts_ids_base.isin(all_tissue_genes)
            tissue_counts_filtered = tissue_counts.loc[mask]
        else:
            tissue_counts_filtered = tissue_counts
        summarize_dataset(f"{label} – {tissue}", tissue_counts_filtered, tissue_meta)
        # Report per-task DGE gene set sizes
        if dge_gene_sets and tissue in dge_gene_sets:
            for task, gene_set in sorted(dge_gene_sets[tissue].items()):
                overlap = counts_ids_base.isin(gene_set).sum()
                print(
                    f"         DGE genes ({task}): {len(gene_set):,} in gene list "
                    f"| {overlap:,} present in counts matrix"
                )


def get_condition_series(metadata: pd.DataFrame, shared_ids: Optional[pd.Index] = None) -> pd.Series:
    column = detect_condition_column(metadata)
    if not column:
        return pd.Series(dtype=object)
    series = metadata.set_index("individual_clean")[column]
    if shared_ids is not None:
        series = series.loc[series.index.intersection(shared_ids)]
    series = series[~series.index.duplicated(keep="first")]
    return series.dropna()


def summarize_condition_alignment(blood_series: pd.Series, brain_series: pd.Series) -> None:
    if blood_series.empty or brain_series.empty:
        print("  Condition alignment unavailable (missing labels in one of the datasets).")
        return
    table = (
        pd.concat(
            {
                "blood": blood_series.value_counts(dropna=False),
                "brain": brain_series.value_counts(dropna=False),
            },
            axis=1,
        )
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    print("  Condition counts across tissues (blood vs brain):")
    mismatched = False
    for cond, row in table.iterrows():
        match = row["blood"] == row["brain"]
        marker = "" if match else "  <-- mismatch"
        name = "Unknown" if pd.isna(cond) else str(cond)
        print(f"    - {name:<15} blood={row['blood']:>3} | brain={row['brain']:>3}{marker}")
        if not match:
            mismatched = True
    if mismatched:
        print("Counts differ between tissues for the labels marked above.")
    else:
        print("Blood and brain share identical counts for every label.")


def summarize_intersection(
    blood_counts: pd.DataFrame,
    brain_counts: pd.DataFrame,
    blood_metadata: pd.DataFrame,
    brain_metadata: pd.DataFrame,
    label: str = "Intersection",
) -> None:
    """Print intersection summary between blood and a single brain counts matrix."""
    shared_ids = blood_counts.columns.intersection(brain_counts.columns)
    shared_genes = blood_counts.index.intersection(brain_counts.index)
    print(f"\n[{label}] Shared individuals: {len(shared_ids):,}")
    if shared_ids.empty:
        print("  No common individuals between blood and brain datasets.")
        return
    print(f"  Blood matrix: {blood_counts.shape[0]:,} genes x {len(shared_ids):,} shared samples")
    print(f"  Brain matrix: {brain_counts.shape[0]:,} genes x {len(shared_ids):,} shared samples")
    print(f"  Shared genes: {len(shared_genes):,}")
    blood_shared_meta = drop_duplicate_individuals(
        blood_metadata[blood_metadata["individual_clean"].isin(shared_ids)]
    )
    brain_shared_meta = drop_duplicate_individuals(
        brain_metadata[brain_metadata["individual_clean"].isin(shared_ids)]
    )
    if not blood_shared_meta.empty:
        print("  Blood shared cohort conditions/age:")
        summarize_conditions("    shared blood", blood_shared_meta)
        summarize_age("    shared blood", blood_shared_meta)
    if not brain_shared_meta.empty:
        print(f"  Brain ({label}) shared cohort conditions/age:")
        summarize_conditions(f"    shared brain ({label})", brain_shared_meta)
        summarize_age(f"    shared brain ({label})", brain_shared_meta)
    blood_series = get_condition_series(blood_metadata, shared_ids)
    brain_series = get_condition_series(brain_metadata, shared_ids)
    summarize_condition_alignment(blood_series, brain_series)


def summarize_intersection_by_tissue(
    blood_counts: pd.DataFrame,
    brain_counts: pd.DataFrame,
    blood_metadata: pd.DataFrame,
    brain_metadata: pd.DataFrame,
    tissue_col: str = "tissue",
    dge_gene_sets: Optional[dict[str, dict[str, set[str]]]] = None,
) -> None:
    """Run summarize_intersection for each brain tissue separately, then combined."""
    # Combined first
    summarize_intersection(blood_counts, brain_counts, blood_metadata, brain_metadata,
                           label="Intersection (all brain)")
    if tissue_col not in brain_metadata.columns:
        return
    tissues = sorted(brain_metadata[tissue_col].dropna().unique())
    brain_ids_base = brain_counts.index.str.split(".").str[0]
    for tissue in tissues:
        tissue_brain_meta = brain_metadata[brain_metadata[tissue_col] == tissue].copy()
        tissue_individuals = tissue_brain_meta["individual_clean"].tolist()
        available_cols = [c for c in tissue_individuals if c in brain_counts.columns]
        if not available_cols:
            print(f"\n[Intersection Blood × Brain – {tissue}] No matching columns in brain counts.")
            continue
        tissue_brain_counts = brain_counts[available_cols]
        # Filter rows to DGE genes for this tissue
        if dge_gene_sets and tissue in dge_gene_sets:
            all_tissue_genes: set[str] = set()
            for gene_set in dge_gene_sets[tissue].values():
                all_tissue_genes |= gene_set
            mask = brain_ids_base.isin(all_tissue_genes)
            tissue_brain_counts = tissue_brain_counts.loc[mask]
        summarize_intersection(
            blood_counts,
            tissue_brain_counts,
            blood_metadata,
            tissue_brain_meta,
            label=f"Intersection Blood × Brain – {tissue}",
        )


def drop_duplicate_individuals(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return metadata with a single row per individual (keep first occurrence)."""
    if "individual_clean" not in metadata.columns:
        return metadata
    deduped = metadata.loc[~metadata["individual_clean"].duplicated(keep="first")].copy()
    return deduped


def summarize_blood_timing(metadata: pd.DataFrame, output_html: Optional[Path] = None) -> pd.DataFrame:
    """Analyse how long before death (and before final diagnosis) blood samples were drawn.

    ROSMAP blood samples are drawn at each participant's last clinical visit
    (``dcfdx_lv`` = diagnosis at last visit).  Because ages above 90 are
    privacy-capped at 95 in the public release, a small number of records
    show a negative gap; those are flagged rather than dropped.

    Parameters
    ----------
    metadata:
        Blood metadata DataFrame (must contain ``age_at_exam_num``,
        ``age_death_num``, ``condition``, ``dcfdx_lv``).
    output_html:
        If given, save an interactive Plotly figure to this path.

    Returns
    -------
    DataFrame with per-participant timing added as new columns.
    """
    import warnings
    warnings.filterwarnings("ignore")

    DCFDX_MAP = {1: "NCI", 2: "MCI", 3: "MCI+other", 4: "AD dementia", 5: "other dementia"}
    CONDITION_ORDER = ["AD", "MCI", "CONTROL"]
    COLOR_MAP = {"AD": "#d62728", "MCI": "#ff7f0e", "CONTROL": "#1f77b4"}

    df = metadata.copy()
    df["years_before_death"] = df["age_death_num"] - df["age_at_exam_num"]
    df["dcfdx_label"] = df["dcfdx_lv"].map(DCFDX_MAP)

    # ── Textual summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Blood sample timing analysis")
    print("(age_at_exam_num = age at last clinical visit / blood draw)")
    print("=" * 60)

    print("\nDiagnosis at blood draw (dcfdx_lv) vs classifier label:")
    print(pd.crosstab(df["condition"], df["dcfdx_label"]).to_string())

    print("\nAge at blood draw by condition:")
    print(df.groupby("condition")["age_at_exam_num"].describe().round(1).to_string())

    print("\nYears from blood draw to death by condition:")
    timing_stats = df.groupby("condition")["years_before_death"].agg(
        ["mean", "median", "std", "min", "max"]
    ).round(2)
    print(timing_stats.to_string())

    print("\nNote: negative values (~2 records) reflect the 90+ age-capping artefact in ROSMAP.")

    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        sub_pos = sub[sub["years_before_death"] >= 0]
        n = len(sub_pos)
        if n == 0:
            continue
        w0  = (sub_pos["years_before_death"] == 0).sum()
        w2  = (sub_pos["years_before_death"] <= 2).sum()
        w5  = (sub_pos["years_before_death"] <= 5).sum()
        w10 = (sub_pos["years_before_death"] <= 10).sum()
        print(
            f"\n{cond} (n={n}, excluding age-cap artefacts):\n"
            f"  blood drawn same year as death   : {w0:>3} ({w0/n:.1%})\n"
            f"  blood drawn within  2 yrs of death: {w2:>3} ({w2/n:.1%})\n"
            f"  blood drawn within  5 yrs of death: {w5:>3} ({w5/n:.1%})\n"
            f"  blood drawn within 10 yrs of death: {w10:>3} ({w10/n:.1%})"
        )

    # ── Plotly figure ────────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Clip to ≥ 0 for plotting (removes age-cap artefacts)
        plot_df = df[df["years_before_death"] >= 0].copy()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Years from blood draw to death (by condition)",
                "Age at blood draw (by condition)",
            ),
        )

        for cond in CONDITION_ORDER:
            sub = plot_df[plot_df["condition"] == cond]["years_before_death"]
            fig.add_trace(
                go.Box(
                    y=sub,
                    name=cond,
                    marker_color=COLOR_MAP[cond],
                    boxmean="sd",
                    showlegend=True,
                ),
                row=1, col=1,
            )

        for cond in CONDITION_ORDER:
            sub = plot_df[plot_df["condition"] == cond]["age_at_exam_num"]
            fig.add_trace(
                go.Box(
                    y=sub,
                    name=cond,
                    marker_color=COLOR_MAP[cond],
                    boxmean="sd",
                    showlegend=False,
                ),
                row=1, col=2,
            )

        fig.update_layout(
            title_text=(
                "ROSMAP Blood Sample Timing<br>"
                "<sup>Samples drawn at last clinical visit (dcfdx_lv). "
                "Age ≥ 90 capped at 95 in public release.</sup>"
            ),
            template="plotly_white",
            height=520,
            width=1000,
            legend_title_text="Condition",
            boxmode="group",
        )
        fig.update_yaxes(title_text="Years before death", row=1, col=1)
        fig.update_yaxes(title_text="Age at blood draw", row=1, col=2)

        if output_html:
            output_html.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_html), include_plotlyjs="cdn")
            print(f"\nTiming plot saved to: {output_html}")
        else:
            fig.show()

    except ImportError:
        print("plotly not available – skipping figure generation.")

    return df


def main() -> None:
    args = parse_args()
    blood_paths, brain_paths = resolve_paths(args)
    blood_metadata = load_metadata(blood_paths.metadata)
    brain_metadata = load_metadata(brain_paths.metadata)
    blood_counts = load_counts(blood_paths.counts, blood_metadata)
    brain_counts = load_counts(brain_paths.counts, brain_metadata)
    dge_gene_sets = load_dge_gene_sets(args.brain_dir)
    print("Blood and brain dataset summary:\n")
    summarize_dataset("Blood", blood_counts, blood_metadata)
    summarize_dataset_by_tissue("Brain", brain_counts, brain_metadata, dge_gene_sets=dge_gene_sets)
    summarize_intersection_by_tissue(blood_counts, brain_counts, blood_metadata, brain_metadata, dge_gene_sets=dge_gene_sets)
    timing_html = BASE_DIR / "files" / "blood_timing_plot.html"
    summarize_blood_timing(blood_metadata, output_html=timing_html)


if __name__ == "__main__":
    main()
