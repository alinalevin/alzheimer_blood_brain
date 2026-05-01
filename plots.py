from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

DEFAULT_CSV = Path(__file__).resolve().parent / "files" / "log_model_comparison.csv"
DEFAULT_TASK_CSV = Path(__file__).resolve().parent / "files" / "task_comparison_metrics.csv"
DEFAULT_HTML = Path(__file__).resolve().parent / "files" / "log_model_comparison_plot.html"
DEFAULT_PNG = Path(__file__).resolve().parent / "files" / "log_model_comparison_plot.png"

# LOO result files – one per (task, tissue) combination
TASK_LOO_FILES = {
    "AD_CONTROL_DLPFC":  Path(__file__).resolve().parent / "files" / "comparison_results_AD_CONTROL_DLPFC.csv",
    "AD_CONTROL_PCC":    Path(__file__).resolve().parent / "files" / "comparison_results_AD_CONTROL_PCC.csv",
    "MCI_CONTROL_DLPFC": Path(__file__).resolve().parent / "files" / "comparison_results_MCI_CONTROL_DLPFC.csv",
    "MCI_CONTROL_PCC":   Path(__file__).resolve().parent / "files" / "comparison_results_MCI_CONTROL_PCC.csv",
}
METRICS = (
    "accuracy",
    "f1",
    "roc_auc",
)


def load_comparison(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"task", "model"}
    for metric in METRICS:
        expected_cols.update({f"blood_{metric}", f"brain_{metric}"})
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return df


def load_task_comparison(csv_path: Path) -> pd.DataFrame:
    """Load task comparison CSV (no model column, just task-level metrics)."""
    df = pd.read_csv(csv_path)
    expected_cols = {"task"}
    for metric in METRICS:
        expected_cols.update({f"blood_{metric}", f"brain_{metric}"})
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    # Add a dummy model column for compatibility with existing code
    df["model"] = "Leave One Out"
    return df


def load_loo_comparison(task_loo_files: dict) -> pd.DataFrame:
    """Load per-subject LOO results and compute per-(task, tissue) aggregate metrics.

    Returns a DataFrame with columns: task, tissue, model, blood_accuracy,
    blood_f1, blood_roc_auc, brain_accuracy, brain_f1, brain_roc_auc.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    rows = []
    for key, path in task_loo_files.items():
        if not path.exists():
            print(f"  Missing LOO file: {path}")
            continue
        df = pd.read_csv(path, index_col=0)
        # key is like AD_CONTROL_DLPFC or MCI_CONTROL_PCC
        parts = key.rsplit("_", 1)
        task = parts[0]   # AD_CONTROL or MCI_CONTROL
        tissue = parts[1] # DLPFC or PCC

        y_true = df["true_label_num"].astype(int)
        row = {"task": task, "tissue": tissue, "model": "Leave One Out"}
        for src, pred_col, prob_col in [
            ("blood", "prediction_blood", "probability_blood"),
            ("brain", "prediction_brain", "probability_brain"),
        ]:
            y_pred = df[pred_col].astype(int)
            row[f"{src}_accuracy"] = accuracy_score(y_true, y_pred)
            row[f"{src}_f1"] = f1_score(y_true, y_pred, zero_division=0)
            try:
                row[f"{src}_roc_auc"] = roc_auc_score(y_true, df[prob_col].astype(float))
            except Exception:
                row[f"{src}_roc_auc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    id_vars = ["task", "model"]
    if "tissue" in df.columns:
        id_vars.append("tissue")
    records = []
    for metric in METRICS:
        metric_df = df.melt(
            id_vars=id_vars,
            value_vars=[f"blood_{metric}", f"brain_{metric}"],
            var_name="source",
            value_name="score",
        )
        metric_df["metric"] = metric.replace("_", " ").title()
        metric_df["source"] = metric_df["source"].str.split("_").str[0].str.title()
        records.append(metric_df)
    return pd.concat(records, ignore_index=True)


def select_best_model(df: pd.DataFrame, task: str, source: str = "blood") -> str:
    """Select the best model based on average performance across metrics.

    Args:
        df: Original comparison dataframe
        task: Task name (e.g., 'AD_CONTROL')
        source: 'blood' or 'brain'

    Returns:
        Name of the best performing model
    """
    task_df = df[df["task"] == task].copy()

    # Calculate average score across selected metrics
    metric_cols = [f"{source}_{m}" for m in METRICS]
    task_df["avg_score"] = task_df[metric_cols].mean(axis=1)

    # Return model with highest average
    best_idx = task_df["avg_score"].idxmax()
    return task_df.loc[best_idx, "model"]


def make_figure(long_df: pd.DataFrame, single_task: Optional[str] = None, df: Optional[pd.DataFrame] = None, tissue: Optional[str] = None):
    """Create a bar plot comparing blood vs brain model performance.

    Args:
        long_df: DataFrame in long format with columns: task, model, source, metric, score
        single_task: If provided, filter to only this task and create a single-row plot
        df: Original dataframe (needed for best model selection)
        tissue: Optional tissue label (DLPFC or PCC) to include in the title
    """
    # Format task names for better display
    def format_task_name(task: str) -> str:
        """Convert AD_CONTROL to 'AD vs Control', MCI_CONTROL to 'MCI vs Control'"""
        parts = task.split('_')
        if len(parts) == 2:
            return f"{parts[0]} vs {parts[1].title()}"
        return task

    long_df = long_df.copy()
    long_df["task_original"] = long_df["task"]  # Keep original for filtering
    long_df["task"] = long_df["task"].apply(format_task_name)

    tissue_label = f" – Blood vs {tissue}" if tissue else ""

    if single_task:
        formatted_task = format_task_name(single_task)

        # Find best model for blood
        best_model = select_best_model(df, single_task, "blood")
        print(f"Best model for {formatted_task}{tissue_label}: {best_model}")

        # Filter to only the best model and the specified task
        plot_df = long_df[
            (long_df["task"] == formatted_task) &
            (long_df["model"] == best_model)
        ].copy()

        if plot_df.empty:
            raise ValueError(f"No data found for task: {single_task}, model: {best_model}")

        # Source label: rename "Brain" → tissue name when known
        if tissue:
            plot_df["source"] = plot_df["source"].replace({"Brain": tissue})

        # Create simple bar plot with metrics on x-axis
        source_order = ["Blood", tissue if tissue else "Brain"]
        color_map = {source_order[0]: "#ffb3ba", source_order[1]: "#bae1ff"}
        fig = px.bar(
            plot_df,
            x="metric",
            y="score",
            color="source",
            barmode="group",
            category_orders={"source": source_order},
            color_discrete_map=color_map,
            height=500,
            width=800,
            text="score",
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_yaxes(range=[0, 1.05], title="Score")
        fig.update_xaxes(title="Metric")
        fig.update_layout(
            title=f"{formatted_task}{tissue_label}<br>{best_model}",
            legend_title_text="Dataset",
            bargap=0.15,
            template="plotly_white",
            font=dict(size=14),
        )
    else:
        # Multi-task layout: rename "Brain" source to tissue name
        if tissue:
            long_df["source"] = long_df["source"].replace({"Brain": tissue})
        source_order = ["Blood", tissue if tissue else "Brain"]
        color_map = {source_order[0]: "#d62728", source_order[1]: "#1f77b4"}
        fig = px.bar(
            long_df,
            x="model",
            y="score",
            color="source",
            barmode="group",
            facet_row="task",
            facet_col="metric",
            category_orders={"source": source_order},
            color_discrete_map=color_map,
            height=600,
        )
        fig.update_yaxes(matches=None, range=[0, 1])
        title = f"Blood vs {tissue} Model Performance" if tissue else "Blood vs Brain Model Performance"
        fig.update_layout(
            title=title,
            legend_title_text="Dataset",
            bargap=0.12,
            template="plotly_white",
        )
    return fig


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot blood vs brain model comparison")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to log_model_comparison.csv")
    parser.add_argument("--use-task-csv", action="store_true", help="Use task_comparison_metrics.csv instead of log_model_comparison.csv")
    parser.add_argument("--html-out", type=Path, default=DEFAULT_HTML, help="Path to save interactive HTML plot")
    parser.add_argument("--png-out", type=Path, default=None, help="Optional path to save a static PNG (requires kaleido)")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive figure display")
    parser.add_argument("--task", type=str, default=None, help="Filter to specific task (e.g., AD_CONTROL)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    count_plots()

    output_dir = Path(__file__).resolve().parent / "files"

    if args.use_task_csv:
        # Use per-subject LOO files to build separate plots per tissue
        loo_df = load_loo_comparison(TASK_LOO_FILES)
        if loo_df.empty:
            print("No LOO comparison files found.")
            return

        tissues = loo_df["tissue"].unique() if "tissue" in loo_df.columns else [None]
        tasks_in_data = loo_df["task"].unique()

        for tissue in sorted(tissues):
            tissue_df = loo_df[loo_df["tissue"] == tissue].copy() if tissue else loo_df.copy()

            # Determine which tasks to plot
            if args.task:
                if args.task not in tasks_in_data:
                    print(f"Task {args.task} not found for tissue {tissue}. Available: {list(tasks_in_data)}")
                    continue
                plot_tasks = [args.task]
            else:
                plot_tasks = list(tasks_in_data)

            for task in sorted(plot_tasks):
                task_tissue_df = tissue_df[tissue_df["task"] == task].copy()
                if task_tissue_df.empty:
                    continue
                long_df = melt_to_long(task_tissue_df)
                fig = make_figure(long_df, single_task=task, df=task_tissue_df, tissue=tissue)

                tissue_str = f"_{tissue.lower()}" if tissue else ""
                task_str = f"_{task.lower()}"
                html_filename = f"task_comparison{task_str}{tissue_str}_plot.html"
                html_out = output_dir / html_filename
                fig.write_html(html_out, include_plotlyjs="cdn")
                print(f"Plot saved to: {html_out}")

                if args.png_out:
                    png_out = output_dir / html_filename.replace(".html", ".png")
                    fig.write_image(str(png_out))
                if not args.no_show:
                    fig.show()
    else:
        # Log-model comparison (not tissue-split)
        df = load_comparison(args.csv)
        long_df = melt_to_long(df)
        fig = make_figure(long_df, single_task=args.task, df=df)

        if args.html_out == DEFAULT_HTML:
            prefix = "log_model_comparison"
            task_suffix = f"_{args.task.lower()}" if args.task else ""
            html_filename = f"{prefix}{task_suffix}_plot.html"
            html_out = output_dir / html_filename
        else:
            html_out = args.html_out

        fig.write_html(html_out, include_plotlyjs="cdn")
        print(f"Plot saved to: {html_out}")

        if args.png_out:
            fig.write_image(str(args.png_out))
        if not args.no_show:
            fig.show()


def count_plots():
    output_html = "counts_summary.html"

    # -----------------------
    # Data – one row per (dataset, diagnosis)
    # "Paired" now split into Blood×DLPFC and Blood×PCC
    # -----------------------
    data = {
        "Dataset": ["Blood", "Blood", "Blood",
                    "Brain DLPFC", "Brain DLPFC", "Brain DLPFC",
                    "Brain PCC",   "Brain PCC",   "Brain PCC",
                    "Paired Blood×DLPFC", "Paired Blood×DLPFC", "Paired Blood×DLPFC",
                    "Paired Blood×PCC",   "Paired Blood×PCC",   "Paired Blood×PCC"],
        "Diagnosis": ["AD", "MCI", "Control"] * 5,
        "Count": [145, 135, 320,
                  362, 233, 305,
                  146, 121, 176,
                   97,  65,  90,
                   48,  45,  53],
    }
    # Simple summary table
    summary = {
        "Dataset":  ["Blood", "Brain DLPFC", "Brain PCC",
                     "Paired Blood×DLPFC", "Paired Blood×PCC"],
        "AD":       [145, 362, 146,  97,  48],
        "MCI":      [135, 233, 121,  65,  45],
        "Control":  [320, 305, 176,  90,  53],
        "Total":    [600, 900, 443, 252, 146],
    }
    df_summary = pd.DataFrame(summary)
    df_long = pd.DataFrame(data)
    color_map = {"AD": "#d62728", "MCI": "#ff7f0e", "Control": "#1f77b4"}

    # -----------------------
    # Create subplots layout: 3 pies (blood, DLPFC, PCC) + bar + counts + pct
    # -----------------------
    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
            [{"type": "domain"}, {"type": "domain"}, {"type": "xy"}],
            [{"type": "xy"},     {"type": "xy"},     {"type": "xy"}],
        ],
        subplot_titles=[
            "Blood (Monocytes)",
            "Brain – DLPFC",
            "Brain – PCC",
            "Paired Blood × DLPFC",
            "Paired Blood × PCC",
            "Total Samples per Dataset",
            "Diagnosis Counts",
            "Diagnosis Counts (Paired)",
            "Diagnosis Distribution (%)",
        ],
    )

    # Pie – Blood
    fig.add_trace(
        px.pie(names=["AD", "MCI", "Control"], values=[145, 135, 320], hole=0.3)
          .update_traces(textinfo="label+value", textfont_size=14,
                         marker_colors=["#d62728", "#ff7f0e", "#1f77b4"]).data[0],
        row=1, col=1,
    )
    # Pie – Brain DLPFC
    fig.add_trace(
        px.pie(names=["AD", "MCI", "Control"], values=[362, 233, 305], hole=0.3)
          .update_traces(textinfo="label+value", textfont_size=14,
                         marker_colors=["#d62728", "#ff7f0e", "#1f77b4"]).data[0],
        row=1, col=2,
    )
    # Pie – Brain PCC
    fig.add_trace(
        px.pie(names=["AD", "MCI", "Control"], values=[146, 121, 176], hole=0.3)
          .update_traces(textinfo="label+value", textfont_size=14,
                         marker_colors=["#d62728", "#ff7f0e", "#1f77b4"]).data[0],
        row=1, col=3,
    )
    # Pie – Paired Blood×DLPFC
    fig.add_trace(
        px.pie(names=["AD", "MCI", "Control"], values=[97, 65, 90], hole=0.3)
          .update_traces(textinfo="label+value", textfont_size=14,
                         marker_colors=["#d62728", "#ff7f0e", "#1f77b4"]).data[0],
        row=2, col=1,
    )
    # Pie – Paired Blood×PCC
    fig.add_trace(
        px.pie(names=["AD", "MCI", "Control"], values=[48, 45, 53], hole=0.3)
          .update_traces(textinfo="label+value", textfont_size=14,
                         marker_colors=["#d62728", "#ff7f0e", "#1f77b4"]).data[0],
        row=2, col=2,
    )

    # Bar – Total samples
    fig.add_trace(
        px.bar(df_summary, x="Dataset", y="Total",
               color_discrete_sequence=["#636efa"])
          .update_traces(text=df_summary["Total"], textposition="outside").data[0],
        row=2, col=3,
    )

    # Grouped bar – all datasets, counts
    for diagnosis in ["AD", "MCI", "Control"]:
        sub = df_long[df_long["Diagnosis"] == diagnosis]
        fig.add_trace(
            px.bar(sub, x="Dataset", y="Count", color="Diagnosis",
                   color_discrete_map=color_map)
              .update_traces(showlegend=True, name=diagnosis).data[0],
            row=3, col=1,
        )

    # Grouped bar – paired only, counts
    paired_long = df_long[df_long["Dataset"].str.startswith("Paired")]
    for diagnosis in ["AD", "MCI", "Control"]:
        sub = paired_long[paired_long["Diagnosis"] == diagnosis]
        fig.add_trace(
            px.bar(sub, x="Dataset", y="Count", color="Diagnosis",
                   color_discrete_map=color_map)
              .update_traces(showlegend=False, name=diagnosis).data[0],
            row=3, col=2,
        )

    # Grouped bar – all datasets, percentage
    df_long["Percentage"] = df_long.groupby("Dataset")["Count"].transform(
        lambda x: x / x.sum() * 100
    )
    for diagnosis in ["AD", "MCI", "Control"]:
        sub = df_long[df_long["Diagnosis"] == diagnosis]
        fig.add_trace(
            px.bar(sub, x="Dataset", y="Percentage", color="Diagnosis",
                   color_discrete_map=color_map)
              .update_traces(showlegend=False, name=diagnosis).data[0],
            row=3, col=3,
        )

    fig.update_layout(
        height=1400,
        width=1300,
        title_text="ROSMAP Dataset Composition: Blood, Brain (DLPFC / PCC), and Paired Samples",
        showlegend=True,
        legend_title_text="Diagnosis",
    )
    fig.update_yaxes(title_text="Samples",    row=2, col=3)
    fig.update_yaxes(title_text="Count",      row=3, col=1)
    fig.update_yaxes(title_text="Count",      row=3, col=2)
    fig.update_yaxes(title_text="Percentage (%)", row=3, col=3)
    fig.update_xaxes(tickangle=30, row=3, col=1)
    fig.update_xaxes(tickangle=30, row=3, col=2)
    fig.update_xaxes(tickangle=30, row=3, col=3)
    fig.update_xaxes(tickangle=30, row=2, col=3)

    fig.write_html(output_html, include_plotlyjs="cdn")

    return output_html


def paired_pie_and_bar_plot():
    output_html = "paired_counts_summary.html"
    color_map = {"AD": "#d62728", "MCI": "#ff7f0e", "Control": "#1f77b4"}

    paired_data = {
        "Blood×DLPFC": {"AD": 97, "MCI": 65, "Control": 90},
        "Blood×PCC":   {"AD": 48, "MCI": 45, "Control": 53},
    }

    # Build long df for bar
    rows = []
    for group, counts in paired_data.items():
        for diag, cnt in counts.items():
            rows.append({"Group": group, "Diagnosis": diag, "Count": cnt})
    df_long = pd.DataFrame(rows)

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "domain"}, {"type": "domain"}, {"type": "xy"}]],
        subplot_titles=("Paired Blood × DLPFC", "Paired Blood × PCC",
                        "Counts by Paired Group"),
    )

    # Pie – Blood×DLPFC
    pie1 = px.pie(names=["AD", "MCI", "Control"], values=[97, 65, 90],
                  color=["AD", "MCI", "Control"], color_discrete_map=color_map, hole=0.3)
    pie1.update_traces(textinfo="label+value", textfont_size=14)
    fig.add_trace(pie1.data[0], row=1, col=1)

    # Pie – Blood×PCC
    pie2 = px.pie(names=["AD", "MCI", "Control"], values=[48, 45, 53],
                  color=["AD", "MCI", "Control"], color_discrete_map=color_map, hole=0.3)
    pie2.update_traces(textinfo="label+value", textfont_size=14)
    fig.add_trace(pie2.data[0], row=1, col=2)

    # Grouped bar
    for diagnosis in ["AD", "MCI", "Control"]:
        sub = df_long[df_long["Diagnosis"] == diagnosis]
        bar = px.bar(sub, x="Group", y="Count", color="Diagnosis",
                     color_discrete_map=color_map, text="Count")
        bar.update_traces(showlegend=True, name=diagnosis,
                          textposition="outside", cliponaxis=False, textfont_size=13)
        fig.add_trace(bar.data[0], row=1, col=3)

    fig.update_layout(
        title_text="Paired Blood + Brain: Diagnosis Distribution and Counts",
        legend_title_text="Diagnosis",
        showlegend=True,
        barmode="group",
        width=1200,
        height=520,
        margin=dict(t=80, b=100, l=60, r=60),
    )
    fig.update_xaxes(tickangle=20, row=1, col=3)
    fig.update_yaxes(title_text="Number of Samples", row=1, col=3)

    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Saved paired pie and bar plot to {output_html}")
    return output_html


def grouped_bar_and_paired_pie_plot():
    output_html = "grouped_bar_and_paired_pie.html"

    color_map = {"AD": "#d62728", "MCI": "#ff7f0e", "Control": "#1f77b4"}

    # All groups including per-tissue pairs
    data = {
        "Group":     ["Blood",  "Blood",  "Blood",
                      "Brain DLPFC", "Brain DLPFC", "Brain DLPFC",
                      "Brain PCC",   "Brain PCC",   "Brain PCC",
                      "Blood×DLPFC", "Blood×DLPFC", "Blood×DLPFC",
                      "Blood×PCC",   "Blood×PCC",   "Blood×PCC"],
        "Diagnosis": ["AD", "MCI", "Control"] * 5,
        "Count":     [145, 135, 320,
                      362, 233, 305,
                      146, 121, 176,
                       97,  65,  90,
                       48,  45,  53],
    }
    df = pd.DataFrame(data)

    # Two pie charts: one per paired group
    paired_groups = ["Blood×DLPFC", "Blood×PCC"]
    paired_counts = {
        "Blood×DLPFC": {"AD": 97,  "MCI": 65, "Control": 90},
        "Blood×PCC":   {"AD": 48,  "MCI": 45, "Control": 53},
    }

    specs = [[{"type": "domain"}, {"type": "domain"}, {"type": "xy"}]]
    fig = make_subplots(
        rows=1, cols=3,
        specs=specs,
        subplot_titles=("Paired Blood × DLPFC", "Paired Blood × PCC",
                        "Diagnosis Counts by Group"),
    )

    for col_idx, group in enumerate(paired_groups, start=1):
        counts = paired_counts[group]
        pie = px.pie(
            names=list(counts.keys()),
            values=list(counts.values()),
            color=list(counts.keys()),
            color_discrete_map=color_map,
            hole=0.3,
        )
        pie.update_traces(textinfo="label+value", textfont_size=14)
        fig.add_trace(pie.data[0], row=1, col=col_idx)

    # Grouped bar for all groups
    group_order = ["Blood", "Brain DLPFC", "Brain PCC", "Blood×DLPFC", "Blood×PCC"]
    for diagnosis in ["AD", "MCI", "Control"]:
        sub = df[df["Diagnosis"] == diagnosis]
        bar = px.bar(
            sub,
            x="Group",
            y="Count",
            color="Diagnosis",
            color_discrete_map=color_map,
            category_orders={"Group": group_order},
            text="Count",
        )
        bar.update_traces(
            showlegend=True,
            name=diagnosis,
            textposition="outside",
            textfont_size=13,
            cliponaxis=False,
        )
        fig.add_trace(bar.data[0], row=1, col=3)

    fig.update_layout(
        title_text="Diagnosis Distribution and Counts by Group",
        legend_title_text="Diagnosis",
        showlegend=True,
        width=1400,
        height=620,
        margin=dict(t=90, b=120, l=60, r=60),
        barmode="group",
    )
    fig.update_xaxes(tickangle=30, row=1, col=3)
    fig.update_yaxes(title_text="Number of Samples", row=1, col=3)

    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Saved grouped bar and paired pie plot to {output_html}")
    return output_html


def bar_counts_by_group():
    """Grouped bar chart: AD / MCI / Control counts for every dataset group."""
    output_html = str(Path(__file__).resolve().parent / "files" / "bar_counts_by_group.html")

    color_map = {"AD": "#d62728", "MCI": "#ff7f0e", "Control": "#1f77b4"}
    group_order = ["Blood", "Brain DLPFC", "Brain PCC", "Blood×DLPFC", "Blood×PCC"]

    data = {
        "Group":     group_order * 3,
        "Diagnosis": (["AD"] * 5) + (["MCI"] * 5) + (["Control"] * 5),
        "Count":     [145, 362, 146,  97,  48,   # AD
                      135, 233, 121,  65,  45,   # MCI
                      320, 305, 176,  90,  53],  # Control
    }
    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x="Group",
        y="Count",
        color="Diagnosis",
        barmode="group",
        color_discrete_map=color_map,
        category_orders={"Group": group_order, "Diagnosis": ["AD", "MCI", "Control"]},
        text="Count",
        labels={"Count": "Number of Samples", "Group": "Dataset"},
        title="Sample Counts per Group and Diagnosis",
        height=550,
        width=900,
    )
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=12)
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Diagnosis",
        bargap=0.20,
        bargroupgap=0.05,
        font=dict(size=13),
        margin=dict(t=80, b=120, l=60, r=60),
        yaxis=dict(range=[0, 420]),
    )
    fig.update_xaxes(tickangle=20)

    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Saved bar counts by group to {output_html}")
    return output_html


if __name__ == "__main__":
    grouped_bar_and_paired_pie_plot()
    paired_pie_and_bar_plot()
    bar_counts_by_group()
    main()
