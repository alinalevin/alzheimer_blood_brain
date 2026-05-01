"""PCA analysis for Blood, Brain-DLPFC, and Brain-PCC datasets.

Loads expression data and metadata, runs PCA, and plots the first two/three
principal components coloured by diagnosis group (AD / MCI / CONTROL).
Figures are saved as interactive HTML files in files/plots/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "files"
BLOOD_DIR = DATA_DIR / "blood"
BRAIN_DIR = DATA_DIR / "brain"
OUT_DIR = DATA_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette (consistent across datasets) ───────────────────────────────
GROUP_COLORS = {
    "AD": "#E03C31",       # red
    "MCI": "#F5A623",      # orange
    "CONTROL": "#4A90D9",  # blue
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_blood() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts_T [samples x genes], metadata) for blood."""
    metadata = pd.read_excel(BLOOD_DIR / "blood_final_residuals_metadata_all.xlsx")
    counts = pd.read_csv(BLOOD_DIR / "blood_r1_clean_all.csv", index_col=0)  # genes x samples

    # Rename specimen → individual
    spec_to_ind = metadata.dropna(subset=["specimenID", "individualID"]).set_index("specimenID")["individualID"]
    counts = counts.rename(columns=spec_to_ind)

    # Deduplicate by keeping first occurrence per individual
    if counts.columns.duplicated().any():
        counts = counts.T.groupby(level=0).first().T

    # Align metadata index to individualID
    metadata = metadata.drop_duplicates(subset=["individualID"])
    metadata = metadata.set_index("individualID")

    # Keep only samples present in both
    common = counts.columns.intersection(metadata.index)
    counts = counts[common]
    metadata = metadata.loc[common]

    return counts.T, metadata  # samples x genes


def load_brain(tissue: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts_T [samples x genes], metadata) for a given brain tissue.

    Parameters
    ----------
    tissue : "DLPFC" or "PCC"
    """
    metadata_all = pd.read_excel(BRAIN_DIR / "brain_metadata_after_preprocess_all.xlsx")
    counts_all = pd.read_csv(BRAIN_DIR / "brain_residuals_all.csv", index_col=0)  # genes x samples

    # Filter metadata to requested tissue
    metadata = metadata_all[metadata_all["tissue"] == tissue].copy()

    spec_col = "specimenID"
    ind_col = "individualID"

    spec_ids = metadata[spec_col].tolist()
    keep = [s for s in spec_ids if s in counts_all.columns]
    counts = counts_all[keep]

    # Rename specimen → individual, dedup
    spec_to_ind = metadata.dropna(subset=[spec_col, ind_col]).set_index(spec_col)[ind_col]
    counts = counts.rename(columns=spec_to_ind)
    if counts.columns.duplicated().any():
        counts = counts.T.groupby(level=0).first().T

    # Align metadata
    metadata = metadata.drop_duplicates(subset=[ind_col]).set_index(ind_col)
    common = counts.columns.intersection(metadata.index)
    counts = counts[common]
    metadata = metadata.loc[common]

    return counts.T, metadata  # samples x genes


def run_pca(X: pd.DataFrame, n_components: int = 10) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Standardise and run PCA. Returns (scores, explained_variance_ratio, pca_obj)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    scores = pca.fit_transform(X_scaled)
    return scores, pca.explained_variance_ratio_, pca


def plot_pca_2d(
    scores: np.ndarray,
    ev_ratio: np.ndarray,
    labels: pd.Series,
    title: str,
    out_path: Path,
) -> go.Figure:
    """Scatter PC1 vs PC2, coloured by diagnosis group."""
    df = pd.DataFrame(
        {
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "Group": labels.values,
            "Sample": labels.index.tolist(),
        }
    )

    category_order = [g for g in ["AD", "MCI", "CONTROL"] if g in df["Group"].unique()]

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Group",
        color_discrete_map=GROUP_COLORS,
        category_orders={"Group": category_order},
        hover_name="Sample",
        hover_data={"Group": True, "PC1": ":.3f", "PC2": ":.3f"},
        title=title,
        labels={
            "PC1": f"PC1 ({ev_ratio[0]*100:.1f}%)",
            "PC2": f"PC2 ({ev_ratio[1]*100:.1f}%)",
        },
        opacity=0.75,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
    fig.update_layout(
        legend_title_text="Diagnosis",
        title_font_size=16,
        width=750,
        height=600,
    )
    fig.write_html(str(out_path))
    print(f"Saved 2-D PCA plot -> {out_path}")
    return fig


def plot_pca_3d(
    scores: np.ndarray,
    ev_ratio: np.ndarray,
    labels: pd.Series,
    title: str,
    out_path: Path,
) -> go.Figure:
    """Scatter PC1/PC2/PC3, coloured by diagnosis group."""
    df = pd.DataFrame(
        {
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "PC3": scores[:, 2],
            "Group": labels.values,
            "Sample": labels.index.tolist(),
        }
    )

    category_order = [g for g in ["AD", "MCI", "CONTROL"] if g in df["Group"].unique()]

    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Group",
        color_discrete_map=GROUP_COLORS,
        category_orders={"Group": category_order},
        hover_name="Sample",
        title=title,
        labels={
            "PC1": f"PC1 ({ev_ratio[0]*100:.1f}%)",
            "PC2": f"PC2 ({ev_ratio[1]*100:.1f}%)",
            "PC3": f"PC3 ({ev_ratio[2]*100:.1f}%)",
        },
        opacity=0.75,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        legend_title_text="Diagnosis",
        title_font_size=16,
    )
    fig.write_html(str(out_path))
    print(f"Saved 3-D PCA plot -> {out_path}")
    return fig


def plot_scree(
    ev_ratio: np.ndarray,
    title: str,
    out_path: Path,
    max_components: int = 10,
) -> go.Figure:
    """Scree plot of explained variance ratio."""
    n = min(max_components, len(ev_ratio))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"PC{i+1}" for i in range(n)],
            y=ev_ratio[:n] * 100,
            marker_color="#4A90D9",
            name="Variance explained",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[f"PC{i+1}" for i in range(n)],
            y=np.cumsum(ev_ratio[:n]) * 100,
            mode="lines+markers",
            marker=dict(color="#E03C31"),
            name="Cumulative",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        template="plotly_white",
        legend=dict(x=0.75, y=0.5),
        width=700,
        height=450,
    )
    fig.write_html(str(out_path))
    print(f"Saved scree plot -> {out_path}")
    return fig


def analyse_dataset(
    counts_T: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset_name: str,
    n_components: int = 10,
) -> list[go.Figure]:
    """Run full PCA analysis for one dataset, save individual plots, and return figures."""
    print(f"\n{'='*60}")
    print(f"PCA analysis: {dataset_name}")
    print(f"  Samples: {counts_T.shape[0]}")
    print(f"  Genes:   {counts_T.shape[1]}")
    print(f"  Groups:  {metadata['condition'].value_counts().to_dict()}")
    print(f"{'='*60}")

    scores, ev_ratio, _ = run_pca(counts_T, n_components=n_components)

    safe_name = dataset_name.replace(" ", "_")
    labels = metadata["condition"]

    figs = []

    # 2-D scatter
    fig2d = plot_pca_2d(
        scores, ev_ratio, labels,
        title=f"PCA – {dataset_name} (PC1 vs PC2)",
        out_path=OUT_DIR / f"pca_2d_{safe_name}.html",
    )
    figs.append(("2-D PCA", fig2d))

    # 3-D scatter
    if scores.shape[1] >= 3:
        fig3d = plot_pca_3d(
            scores, ev_ratio, labels,
            title=f"PCA – {dataset_name} (PC1 / PC2 / PC3)",
            out_path=OUT_DIR / f"pca_3d_{safe_name}.html",
        )
        figs.append(("3-D PCA", fig3d))

    # Scree plot
    fig_scree = plot_scree(
        ev_ratio,
        title=f"Scree Plot – {dataset_name}",
        out_path=OUT_DIR / f"pca_scree_{safe_name}.html",
    )
    figs.append(("Scree", fig_scree))

    return figs


# ── combine plots ─────────────────────────────────────────────────────────────

SECTION_ORDER = ["Blood", "Brain-DLPFC", "Brain-PCC"]


def combine_plots(
    sections: dict[str, list[tuple[str, go.Figure]]],
    out_path: Path,
) -> None:
    """Embed all Plotly figures directly into one self-contained HTML file."""

    section_html = ""
    for section_name in SECTION_ORDER:
        figs = sections.get(section_name, [])
        cards = ""
        for plot_type, fig in figs:
            # Restore full size and let Plotly be responsive width-wise
            fig.update_layout(width=None, height=650, margin=dict(l=60, r=40, t=70, b=60))
            inner = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
            # Build with concatenation to avoid f-string conflicts with Plotly JSON braces
            cards = (cards
                + '\n        <div class="plot-card">'
                + '\n          <h3 class="plot-subtitle">' + plot_type + '</h3>'
                + '\n          ' + inner
                + '\n        </div>')

        section_html = (section_html
            + '\n    <section class="dataset-section">'
            + '\n      <h2 class="section-title">' + section_name + '</h2>'
            + '\n      <div class="plot-col">'
            + cards
            + '\n      </div>'
            + '\n    </section>'
            + '\n    <hr class="section-divider">')

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    html = (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='UTF-8'>\n"
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
        "  <title>PCA Analysis - Blood, Brain-DLPFC, Brain-PCC</title>\n"
        "  <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n"
        "  <style>\n"
        "    body { font-family: 'Segoe UI', Arial, sans-serif; background:#f7f8fa; margin:0; padding:0 0 40px 0; color:#2d2d2d; }\n"
        "    header { background:#2c3e50; color:white; padding:28px 40px 20px 40px; margin-bottom:30px; }\n"
        "    header h1 { margin:0 0 6px 0; font-size:1.8rem; }\n"
        "    header p  { margin:0; font-size:0.95rem; opacity:0.85; }\n"
        "    .dataset-section { max-width:1400px; margin:0 auto 10px auto; padding:0 30px; }\n"
        "    .section-title { font-size:1.4rem; color:#2c3e50; border-left:5px solid #4A90D9; padding-left:12px; margin:24px 0 16px 0; }\n"
        "    .plot-col { display:flex; flex-direction:column; gap:28px; }\n"
        "    .plot-card { background:white; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); padding:20px; }\n"
        "    .plot-subtitle { margin:0 0 8px 0; font-size:1rem; color:#555; text-align:center; }\n"
        "    .plot-card .plotly-graph-div { width:100% !important; }\n"
        "    .section-divider { max-width:1400px; margin:30px auto; border:none; border-top:2px solid #dde; }\n"
        "    footer { text-align:center; color:#999; font-size:0.82rem; margin-top:30px; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <header>\n"
        "    <h1>PCA Analysis - Diagnosis Group Separation</h1>\n"
        "    <p>Principal Component Analysis on gene expression data for Blood, Brain-DLPFC, and Brain-PCC datasets. "
        "Each point is one individual, coloured by diagnosis group (AD / MCI / CONTROL).</p>\n"
        "  </header>\n"
        + section_html +
        "\n  <footer>Generated " + timestamp + "</footer>\n"
        "</body>\n"
        "</html>"
    )

    out_path.write_text(html, encoding="utf-8")
    print("Combined report saved -> " + str(out_path))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading Blood dataset…")
    blood_X, blood_meta = load_blood()
    blood_figs = analyse_dataset(blood_X, blood_meta, "Blood")

    print("\nLoading Brain-DLPFC dataset…")
    dlpfc_X, dlpfc_meta = load_brain("DLPFC")
    dlpfc_figs = analyse_dataset(dlpfc_X, dlpfc_meta, "Brain_DLPFC")

    print("\nLoading Brain-PCC dataset…")
    pcc_X, pcc_meta = load_brain("PCC")
    pcc_figs = analyse_dataset(pcc_X, pcc_meta, "Brain_PCC")

    print("\nCombining all plots into a single HTML report…")
    combine_plots(
        sections={
            "Blood":      blood_figs,
            "Brain-DLPFC": dlpfc_figs,
            "Brain-PCC":  pcc_figs,
        },
        out_path=OUT_DIR / "pca_combined.html",
    )

    print(f"\nAll PCA plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

