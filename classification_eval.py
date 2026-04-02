import pandas as pd
import numpy as np
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import warnings
from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import roc_curve, auc
import matplotlib
from datetime import datetime

matplotlib.use("Agg")


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings(action="ignore")

from sklearn.metrics import (confusion_matrix, classification_report, matthews_corrcoef, f1_score,
                             recall_score, precision_score,roc_auc_score, accuracy_score)
from sklearn.model_selection import GridSearchCV
from matplotlib      import pyplot as plt
from IPython.display import display
import itertools
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet


def _metadata_id_column(metadata: pd.DataFrame) -> str:
    return "individualID" if "individualID" in metadata.columns else "specimenID"


def _metadata_ids(metadata: pd.DataFrame) -> pd.Series:
    column = _metadata_id_column(metadata)
    if column in metadata.columns:
        ids = metadata[column]
    else:
        ids = metadata.index.to_series()
    return ids.astype(str)


def get_feature_matrix(counts, metadata, genes_df):
    # create train feature matrix - filter the counts by significant genes
    # depending on the condition
    feature_matrix = counts[genes_df.index]
    feature_matrix.reset_index(inplace=True, drop=False)
    id_column = _metadata_id_column(metadata)
    feature_matrix.rename(columns={'index': id_column}, inplace=True)
    feature_matrix.set_index(id_column, inplace=True, drop=True)
    return feature_matrix

def get_x_train(X, train_metadata, remove_condition, genes_df):
    train_metadata = train_metadata[train_metadata['condition'] != remove_condition]
    ids = _metadata_ids(train_metadata)
    available = [idx for idx in ids if idx in X.index]
    train_counts = X.loc[available]
    x_train = get_feature_matrix(train_counts, train_metadata)
    return x_train

def get_x_test(X, test_metadata, remove_condition):
    test_metadata = test_metadata[test_metadata['condition'] != remove_condition]
    ids = _metadata_ids(test_metadata)
    available = [idx for idx in ids if idx in X.index]
    test_counts = X.loc[available]
    x_test = get_feature_matrix(test_counts, test_metadata)
    return x_test

def get_ad_control_y(metadata):
    y = metadata['condition']
    y = y.replace({'CONTROL': 0, 'AD': 1 })
    # convert to 1-D array
    y = y.values.ravel()
    return y

def get_mci_control_y(metadata):
    y = metadata['condition']
    y = y.replace({'CONTROL': 0, 'MCI': 1 })
    # convert to 1-D array
    y = y.values.ravel()
    return y

def get_ad_mci_y(metadata):
    y = metadata['condition']
    y = y.replace({'MCI': 0, 'AD': 1 })
    # convert to 1-D array
    y = y.values.ravel()
    return y

def get_train_counts_by_condition(X, train_metadata, remove_condition, genes_df):

    train_metadata = train_metadata[train_metadata['condition'] != remove_condition]
    ids = _metadata_ids(train_metadata)
    available = [idx for idx in ids if idx in X.columns]
    train_counts = X[available].T
    # filter the genes that are significant in gene_df
    # the genes are the columns in the counts
    train_counts = train_counts[train_counts.columns.intersection(genes_df.index)]
    return train_counts

def get_test_counts_by_condition(X, test_metadata ,remove_condition, genes_df):
    test_metadata = test_metadata[test_metadata['condition'] != remove_condition]
    ids = _metadata_ids(test_metadata)
    available = [idx for idx in ids if idx in X.columns]
    test_counts = X[available].T
    # filter the genes that are significant in gene_df
    # the genes are the columns in the counts
    test_counts = test_counts[test_counts.columns.intersection(genes_df.index)]
    return test_counts

def get_ad_control_train_test_data(X, train_metadata, test_metadata, genes, condition_to_remove="MCI"  ):
     # read train data
     train_ad_control_metadata = train_metadata.loc[train_metadata['condition']!=condition_to_remove]
     train_ad_control_counts = get_train_counts_by_condition(X,
                                                             train_ad_control_metadata,
                                                             condition_to_remove,
                                                             genes)
     train_common = train_ad_control_counts.index.intersection(train_ad_control_metadata.index)
     train_ad_control_counts = train_ad_control_counts.loc[train_common]
     train_ad_control_metadata = train_ad_control_metadata.loc[train_common]
     # read test data
     test_ad_control_metadata = test_metadata.loc[test_metadata['condition']!=condition_to_remove]
     test_ad_control_counts = get_test_counts_by_condition(X, test_ad_control_metadata,
                                                           condition_to_remove,
                                                           genes)
     test_common = test_ad_control_counts.index.intersection(test_ad_control_metadata.index)
     test_ad_control_counts = test_ad_control_counts.loc[test_common]
     test_ad_control_metadata = test_ad_control_metadata.loc[test_common]
     # convert the x_train to numerical
     x_train = train_ad_control_counts.apply(pd.to_numeric)
     y_train = get_ad_control_y(train_ad_control_metadata)
     x_test = test_ad_control_counts.apply(pd.to_numeric)
     y_test = get_ad_control_y(test_ad_control_metadata)
     return x_train, y_train, x_test, y_test

def get_mci_control_train_test_data(X, train_metadata, test_metadata, genes, condition_to_remove="AD" ):
     # read train data
     train_mci_control_metadata = train_metadata.loc[train_metadata['condition']!=condition_to_remove]
     train_mci_control_counts = get_train_counts_by_condition(X,
                                                              train_mci_control_metadata,
                                                              condition_to_remove,
                                                              genes)
     train_common = train_mci_control_counts.index.intersection(train_mci_control_metadata.index)
     train_mci_control_counts = train_mci_control_counts.loc[train_common]
     train_mci_control_metadata = train_mci_control_metadata.loc[train_common]
     # read test data
     test_mci_control_metadata = test_metadata.loc[test_metadata['condition']!=condition_to_remove]
     test_mci_control_counts = get_test_counts_by_condition(X,
                                                            test_mci_control_metadata,
                                                            condition_to_remove,
                                                            genes)
     test_common = test_mci_control_counts.index.intersection(test_mci_control_metadata.index)
     test_mci_control_counts = test_mci_control_counts.loc[test_common]
     test_mci_control_metadata = test_mci_control_metadata.loc[test_common]
     # convert the x_train to numerical
     x_train = train_mci_control_counts.apply(pd.to_numeric)
     y_train = get_mci_control_y(train_mci_control_metadata)
     x_test = test_mci_control_counts.apply(pd.to_numeric)
     y_test = get_mci_control_y(test_mci_control_metadata)
     return x_train, y_train, x_test, y_test

#### Feature_importance ###################################################
def get_rf_feature_importance(rf_model, x_train, genes_df, model_type):
    importance = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': importance,
    }).sort_values('Importance', ascending=False)
    feature_importance_df = feature_importance_df.merge(
        genes_df[['symbol']], left_on='Feature', right_index=True
    )
    output_dir = get_output_dir()
    feature_importance_df.to_csv(output_dir / f"feature_importance_rf_{model_type}.csv")
    return feature_importance_df


def plot_rf_feature_importance(feature_importance_df, model_type):
    """Create interactive Plotly bar chart for Random Forest feature importance."""
    top_features = feature_importance_df.head(50)

    fig = go.Figure(data=go.Bar(
        x=top_features['Importance'],
        y=top_features['symbol'],
        orientation='h',
        marker=dict(color='lightblue'),
        text=top_features['Importance'],
        texttemplate='%{text:.4f}',
        textposition='outside',
        hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))

    # Format task name
    task_display = model_type.replace("AD_CONTROL", "AD vs Control").replace("MCI_CONTROL", "MCI vs Control")

    fig.update_layout(
        title=f'Feature Importance for {task_display}<br>Model: Balanced Random Forest',
        xaxis_title='Feature Importance',
        yaxis_title='Feature',
        height=max(600, len(top_features) * 15),
        width=900,
        font=dict(size=10),
        template="plotly_white",
        yaxis=dict(autorange="reversed")
    )

    output_dir = get_output_dir() / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rf_feature_importance_{model_type}.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"RF feature importance plot saved to: {output_path}")



def plot_all_models_roc_curve(y_test, model_pred_probs, model_type, tissue, path):
    """Create interactive Plotly ROC curve comparing all models."""
    colors = {'RF': 'red', 'XGB': 'green', 'LR': 'blue'}

    traces = []
    for model_name, probs in model_pred_probs.items():
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        model_roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        model_roc_df.to_csv(f"{path}/{tissue}_{model_name}_{model_type}_roc_df.csv", index=False)
        roc_auc = auc(fpr, tpr)

        trace = go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f"{model_name} (AUC = {roc_auc:.2f})",
            line=dict(color=colors[model_name], width=2),
            hovertemplate=f'{model_name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'
        )
        traces.append(trace)

    # Add diagonal line (no skill)
    diagonal_trace = go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='No Skill',
        line=dict(color='grey', width=2, dash='dash'),
        hovertemplate='%{x:.3f}<extra></extra>'
    )
    traces.append(diagonal_trace)

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=f'{tissue} ROC curve for {model_type}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800,
        height=800,
        font=dict(size=14),
        template="plotly_white",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.6, y=0.1)
    )

    output_dir = get_output_dir() / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"roc_all_models_{tissue}_{model_type}.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"All models ROC curve saved to: {output_path}")



def feature_selection_elastic_net(x_train, y_train, alpha=0.001, l1_ratio=0.1):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x_train)
    elastic_net_model = ElasticNet(max_iter=10000, alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    elastic_net_model.fit(X_scaled, y_encoded)
    coefficients = elastic_net_model.coef_
    feature_names = x_train.columns
    feature_importance = pd.DataFrame({'ensemble_gene_id': feature_names, 'Coefficient': coefficients})
    feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()
    feature_importance = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)
    feature_importance = feature_importance[feature_importance['Coefficient'] != 0]
    print(f"The number of features selected by ElasticNet : {feature_importance.shape[0]}")
    return feature_importance


def print_train_scores(cv_res):
    print("Evaluation metrics for cross validation of the model (Train data):")
    for key, value in cv_res.items():
        if key.startswith('train_'):
            print(key, value.mean())


def print_test_scores(cv_res):
    print("Evaluation metrics for cross validation of the model (Test data):")
    for key, value in cv_res.items():
        if key.startswith('test_'):
            print(key, value.mean())


def get_wrong_case_ids(y_pred, x_test, y_test):
    y_pred = pd.Series(y_pred, index=x_test.index)
    y_test = pd.Series(y_test, index=x_test.index)
    fp_cases = y_test[(y_test == 0) & (y_pred == 1)]
    fn_cases = y_test[(y_test == 1) & (y_pred == 0)]
    print("False Positive cases:")
    print(fp_cases)
    print("False Negative cases:")
    print(fn_cases)


def get_output_dir():
    # Check for both BRAIN_OUTPUT_DIR and BLOOD_OUTPUT_DIR
    out_dir = os.getenv("BRAIN_OUTPUT_DIR") or os.getenv("BLOOD_OUTPUT_DIR")
    if out_dir:
        path = Path(out_dir)
    else:
        path = Path(__file__).resolve().parent / "files" / "blood"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_plot(filename_prefix):
    output_dir = get_output_dir() / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"{filename_prefix}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    print(f"Saved plot to {file_path}")
    return file_path


def load_model(model_name):
    filename = get_output_dir() / f"{model_name}.pkl"
    loaded_model = joblib.load(filename)
    return loaded_model


def save_model(model, model_name, model_type, x_test, x_train, y_test, y_train):
    output_dir = get_output_dir()
    filename = output_dir / f"{model_name}_{model_type}.pkl"
    joblib.dump(model, filename)
    x_train.to_csv(output_dir / f"x_train_{model_name}_{model_type}.csv")
    x_test.to_csv(output_dir / f"x_test_{model_name}_{model_type}.csv")
    y_test = pd.DataFrame(y_test)
    y_test['y_pred'] = model.predict(x_test)
    y_test['individualID'] = y_test.index
    y_proba = model.predict_proba(x_test)
    y_test['y_prob_0'] = y_proba[:, 1]
    y_test['y_prob_1'] = y_proba[:, 0]
    y_test.to_csv(output_dir / f"y_test_vs_y_pred_{model_name}_{model_type}.csv")
    print(f"Model saved to {filename}")


def plot_lr_feature_importance(model, model_type, x_train, genes_df, tissue):
    """Create interactive Plotly bar chart for Logistic Regression feature importance."""
    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)
    feature_importance_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Coefficient': coefficients,
        'odds_ratios': odds_ratios,
    }).sort_values('Coefficient', ascending=False)
    feature_importance_df = feature_importance_df.merge(
        genes_df[['symbol']], left_on='Feature', right_index=True
    )
    output_dir = get_output_dir()
    feature_importance_df.to_csv(output_dir / f"feature_importance_lr_{model_type}.csv")

    # Create Plotly bar chart
    top_features = feature_importance_df.head(20)

    # Format task name
    task_display = model_type.replace("AD_CONTROL", "AD vs Control").replace("MCI_CONTROL", "MCI vs Control")

    fig = go.Figure(data=go.Bar(
        x=top_features['Coefficient'],
        y=top_features['symbol'],
        orientation='h',
        marker=dict(color='lightcoral'),
        text=top_features['Coefficient'],
        texttemplate='%{text:.2f}',
        textposition='outside',
        hovertemplate='%{y}<br>Coefficient: %{x:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Feature Importance for {task_display}<br>Model: Logistic Regression",
        xaxis_title='Feature Importance (Non zero Coefficient L1 Regularization)',
        yaxis_title='Feature',
        height=600,
        width=900,
        font=dict(size=12),
        template="plotly_white",
        yaxis=dict(autorange="reversed")
    )

    output_dir = get_output_dir() / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lr_feature_importance_{tissue}_{model_type}.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"LR feature importance plot saved to: {output_path}")

    return feature_importance_df


def plot_xgb_feature_importance(model, model_type, genes_df, tissue):
    """Create interactive Plotly bar chart for XGBoost feature importance."""
    importance = model.get_booster().get_score(importance_type="gain", fmap='')
    feature_importance_df = pd.DataFrame({
        'ensembl_gene_id': list(importance.keys()),
        'F_score': list(importance.values()),
    })
    feature_importance_df = feature_importance_df.set_index('ensembl_gene_id', drop=False)
    genes = genes_df.set_index('ensembl_gene_id', drop=False)
    feature_importance_df = feature_importance_df.merge(genes[['symbol']], right_index=True, how='left', left_index=True)
    feature_importance_df = feature_importance_df.sort_values('F_score', ascending=False)
    output_dir = get_output_dir()
    feature_importance_df.to_csv(output_dir / f"feature_importance_xgb_{model_type}.csv")

    # Create Plotly bar chart
    top_features = feature_importance_df.head(20)

    # Format task name
    task_display = model_type.replace("AD_CONTROL", "AD vs Control").replace("MCI_CONTROL", "MCI vs Control")

    fig = go.Figure(data=go.Bar(
        x=top_features['F_score'],
        y=top_features['symbol'],
        orientation='h',
        marker=dict(color='lightgreen'),
        text=top_features['F_score'],
        texttemplate='%{text:.2f}',
        textposition='outside',
        hovertemplate='%{y}<br>F-score: %{x:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Feature Importance for {task_display}<br>Model: XGBoost",
        xaxis_title='Feature Importance (F-score)',
        yaxis_title='',
        height=600,
        width=900,
        font=dict(size=12),
        template="plotly_white",
        yaxis=dict(autorange="reversed")
    )

    output_dir = get_output_dir() / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"xgb_feature_importance_{tissue}_{model_type}.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"XGB feature importance plot saved to: {output_path}")

    return feature_importance_df


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Create an interactive Plotly confusion matrix heatmap."""

    # Create text annotations with counts
    text_annotations = [[str(cm[i, j]) for j in range(cm.shape[1])] for i in range(cm.shape[0])]

    # Determine text color based on cell value (white for dark cells, black for light cells)
    max_val = cm.max()
    threshold = max_val / 2
    text_colors = [['white' if cm[i, j] > threshold else 'black'
                    for j in range(cm.shape[1])]
                   for i in range(cm.shape[0])]

    # Create heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale="Blues",
        showscale=True,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))

    # Add text annotations with dynamic colors
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(
                x=classes[j],
                y=classes[i],
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(size=16, color=text_colors[i][j])
            )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        width=600,
        height=500,
        font=dict(size=14),
        template="plotly_white",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")  # To match sklearn convention
    )

    # Save as HTML
    output_dir = get_output_dir() / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize filename by removing HTML tags and replacing special characters
    sanitized_title = title.replace('<br>', '_').replace(':', '').replace(' ', '_').replace('-', '')
    output_path = output_dir / f"confusion_matrix_{sanitized_title}.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Confusion matrix saved to: {output_path}")



def plot_roc_auc_for_model(model, model_name, x_test, y_test, model_type="AD_CONTROL"):
    """Create interactive Plotly ROC curve."""
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Create ROC curve trace
    roc_trace = go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(color='deepskyblue', width=2),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    )

    # Create diagonal line (no skill)
    diagonal_trace = go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='No Skill',
        line=dict(color='gray', width=2, dash='dash'),
        hovertemplate='%{x:.3f}<extra></extra>'
    )

    fig = go.Figure(data=[roc_trace, diagonal_trace])

    fig.update_layout(
        title=f"ROC Curve for {'AD vs Control' if model_type == 'AD_CONTROL' else 'MCI vs Control' if model_type == 'MCI_CONTROL' else model_type} Classification<br>Model: {model_name}",
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=700,
        font=dict(size=14),
        template="plotly_white",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.6, y=0.1)
    )

    output_dir = get_output_dir() / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"roc_{model_name}_{model_type}.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"ROC curve saved to: {output_path}")



def eval_cls_model(clf, x, y, labels=['0', '1'], model_name=None, task_name=None):
    y_predict = clf.predict(x)
    y_pred_proba = clf.predict_proba(x)[:, 1]  # Get probabilities for positive class
    cm = confusion_matrix(y, y_predict)
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y, y_predict)
        print(f"Accuracy: {accuracy}")
        sensitivity = tp / (tp + fn)
        print(f"Sensitivity (recall) : {sensitivity}")
        precision = precision_score(y, y_predict, pos_label=1, labels=labels)
        print(f"Precision Score : {precision}")
        specificity = tn / (tn + fp)
        print(f"Specificity :  {specificity}\n\n")
        # Use probabilities for ROC AUC (not binary predictions)
        print(f"ROC AUC score: {roc_auc_score(y, y_pred_proba)}")
        print(classification_report(y, y_predict, target_names=labels))
        f1_s = f1_score(y, y_predict, labels=labels, pos_label=1)
        print(f"F1 Score : {f1_s}")
        mcc = matthews_corrcoef(y, y_predict)
        print(f"Matthews correlation coefficient : {mcc}")

        # Format title with model name and task
        if task_name:
            # Format task name: AD_CONTROL -> AD vs Control, MCI_CONTROL -> MCI vs Control
            task_display = task_name.replace("AD_CONTROL", "AD vs Control").replace("MCI_CONTROL", "MCI vs Control")
            title = f"Confusion Matrix - {task_display}"
        else:
            title = "Confusion Matrix"

        if model_name:
            title += f"<br>Model: {model_name}"

        plot_confusion_matrix(cm, labels, title=title)
    return y_predict
