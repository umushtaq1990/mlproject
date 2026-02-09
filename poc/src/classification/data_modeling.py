import os
import mlflow
import mlflow.sklearn
import shap
import pandas as pd
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from pathlib import Path

ROW_ID_COL = "Row_Id"
PLOTS_DIR = Path("plots")
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import yaml
from mlflow.models.signature import infer_signature

def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets from parquet files."""
    data_cfg = config.get("data", {}) if config else {}
    output_dir = data_cfg.get("output_dir", "data")
    train_path_template = data_cfg.get("train_path", "{output_dir}/train-data/train.parquet")
    test_path_template = data_cfg.get("test_path", "{output_dir}/test-data/test.parquet")
    train_path = Path(train_path_template.format(output_dir=output_dir))
    test_path = Path(test_path_template.format(output_dir=output_dir))

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: Dict[str, Any]) -> Any:
    """Train and return a classifier."""
    params = config["modeling"]["model"]["params"]
    params.update(config["modeling"]["model"]["hyperparameters"])
    #model = XGBClassifier(**params)
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return model

def train_model_with_gridsearch(X_train: pd.DataFrame, y_train: pd.Series, config: Dict[str, Any]):
    param_grid = {
        'max_depth': [3, 4, 5],
        'n_estimators': [100, 200]
    }
    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train, y_train)
    return grid_search

# categorize Titanic survival probability into Low (0-0.4), Medium (0.4-0.7) and High (0.7-1)
def categorize_probability(prob):
    if prob <= 0.2:
        return "Low"
    elif prob < 0.8:
        return "Medium"
    else:
        return "High"
    
def predict_model(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    df_pred = X.copy()
    if ROW_ID_COL not in df_pred.columns:
        df_pred = df_pred.reset_index().rename(columns={"index": ROW_ID_COL})
    df_pred["Actual"] = y
    df_pred["Prediction"] = y_pred
    df_pred["Predicted_Probab"] = y_prob
    df_pred["Titanic_Survived"] = df_pred["Predicted_Probab"].apply(categorize_probability)
    return df_pred

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, prefix: str = "") -> Dict[str, float]:
    """Evaluate model and return metrics."""
    valid_mask = y.notna()
    if valid_mask.sum() == 0:
        raise ValueError("No valid target values after dropping NaNs.")
    X_eval = X.loc[valid_mask]
    y_eval = y.loc[valid_mask]
    df_pred = predict_model(model, X_eval, y_eval)
    acc = accuracy_score(y_eval, df_pred["Prediction"])
    auc = roc_auc_score(y_eval, df_pred["Predicted_Probab"])
    report = classification_report(y_eval, df_pred["Prediction"], output_dict=True)
    cm = confusion_matrix(y_eval, df_pred["Prediction"])
    metrics = {
        f"{prefix}accuracy": acc,
        f"{prefix}auc": auc,
        f"{prefix}precision": report["1"]["precision"],
        f"{prefix}recall": report["1"]["recall"],
        f"{prefix}f1": report["1"]["f1-score"]
    }
    return metrics, cm, df_pred

def log_gridsearch_results(grid_search, mlflow):
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_csv = "gridsearch_results.csv"
    results_df.to_csv(results_csv, index=False)
    mlflow.log_artifact(results_csv)

def plot_gridsearch_heatmap(grid_search):
    import matplotlib.pyplot as plt
    import seaborn as sns
    results = pd.DataFrame(grid_search.cv_results_)
    pivot = results.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')
    plt.figure(figsize=(6,4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("GridSearchCV Accuracy Heatmap")
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / "gridsearch_heatmap.png"
    plt.savefig(plot_path)
    plt.close()
    return str(plot_path)

def plot_roc(y_true, y_prob, prefix: str = "") -> str:
    """Plot ROC curve and save to file."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend(loc="lower right")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"{prefix}_roc_curve.png"
    plt.savefig(plot_path)
    plt.close()
    return str(plot_path)

# Add a function to plot the classification report as a table
def plot_classification_report_table(y_true, y_pred, ax=None):
    from sklearn.metrics import classification_report
    if ax is None:
        ax = plt.gca()
    valid_mask = pd.Series(y_true).notna()
    y_true_clean = pd.Series(y_true)[valid_mask]
    y_pred_clean = pd.Series(y_pred).loc[valid_mask]
    if y_true_clean.shape[0] == 0:
        ax.axis('off')
        ax.text(0.5, 0.5, "No valid targets", ha='center', va='center')
        return ax
    report_dict = classification_report(y_true_clean, y_pred_clean, output_dict=True)
    rows = []
    row_labels = []
    for key in report_dict:
        if key in ["accuracy"]:
            continue
        row_labels.append(str(key))
        vals = report_dict[key]
        if isinstance(vals, dict):
            rows.append([f"{vals.get('precision', 0):.2f}", f"{vals.get('recall', 0):.2f}", f"{vals.get('f1-score', 0):.2f}", f"{vals.get('support', 0):.0f}"])
        else:
            rows.append(["", "", f"{vals:.2f}", ""])
    col_labels = ["Precision", "Recall", "F1-score", "Support"]
    ax.axis('off')
    table = ax.table(cellText=rows, rowLabels=row_labels, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    return ax

def plot_confusion_matrix_only(cm, prefix: str = "", ax=None):
    """Plot only the confusion matrix on the given axis (or current axis if None)."""
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{prefix} Confusion Matrix")
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(cm.shape[0])
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([str(i) for i in range(cm.shape[0])])
    ax.set_yticklabels([str(i) for i in range(cm.shape[0])])
    # set label
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax

# Fix the plot_confusion_matrix function to use gridspec and call plot_confusion_matrix_only and plot_classification_report_table
def plot_confusion_matrix(cm, y_true=None, y_pred=None, prefix: str = "") -> str:
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(7, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    plot_confusion_matrix_only(cm, prefix=prefix, ax=ax0)
    if y_true is not None and y_pred is not None:
        valid_mask = pd.Series(y_true).notna()
        y_true = pd.Series(y_true)[valid_mask]
        y_pred = pd.Series(y_pred).loc[valid_mask]
        if y_true.shape[0] > 0:
            ax1 = plt.subplot(gs[1])
            plot_classification_report_table(y_true, y_pred, ax=ax1)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"{prefix}_confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close(fig)
    return str(plot_path)

# 
def plot_cm_predicted_vs_deal_probab(df:pd.DataFrame, 
                                     actual_col = 'Actual', 
                                     model_pred_col = 'Predicted_Probab', 
                                     deal_prob_col = 'Deal_Win_Probab', 
                                     prefix: str='') -> None:
    """
    Create a 3x3 figure (3 rows x 3 cols). Each row:
      - Col 0: confusion matrix for Model (selected by row threshold on model_prob)
      - Col 1: confusion matrix for Deal (if available) for same subset
      - Col 2: numeric KPI summary table for the same subset
    Rows correspond to:
      1) High predicted probabilities: model_prob > 0.85
      2) Low predicted probabilities: model_prob < 0.15
      3) Medium predicted probabilities: 0.15 <= model_prob <= 0.85
    Saved to data/titanic_survived/{prefix}cm_pred_vs_deal.png
    """
    os.makedirs("data/titanic_survived", exist_ok=True)

    if actual_col not in df.columns or model_pred_col not in df.columns:
        raise ValueError(f"Missing required columns: {actual_col} and/or {model_pred_col}")

    # helper to build confusion matrix + accuracy for a subset
    def _cm_and_acc(subdf):
        if subdf is None or subdf.shape[0] == 0:
            return None, None, np.nan, np.nan
        actual = pd.to_numeric(subdf[actual_col], errors='coerce').dropna().astype(int)
        model_proba = pd.to_numeric(subdf[model_pred_col], errors='coerce')
        # align indices where both actual and model_proba exist
        valid_idx = actual.index.intersection(model_proba.dropna().index)
        if len(valid_idx) == 0:
            return None, None, np.nan, np.nan
        actual = actual.loc[valid_idx]
        model_proba = model_proba.loc[valid_idx]
        model_label = (model_proba >= 0.5).astype(int)
        cm_model = confusion_matrix(actual, model_label)
        acc_model = accuracy_score(actual, model_label)
        # deal
        if deal_prob_col in subdf.columns and subdf[deal_prob_col].notna().any():
            deal_proba = pd.to_numeric(subdf.loc[valid_idx, deal_prob_col], errors='coerce')
            if deal_proba.dropna().shape[0] == 0:
                cm_deal, acc_deal = None, np.nan
            else:
                deal_label = (deal_proba >= 0.5).astype(int)
                cm_deal = confusion_matrix(actual.loc[deal_label.index], deal_label)
                acc_deal = accuracy_score(actual.loc[deal_label.index], deal_label)
        else:
            cm_deal, acc_deal = None, np.nan
        return cm_model, cm_deal, acc_model, acc_deal

    def generate_kpi_sum(subdf, kpis:list = ['Exp_Rev','Planned_Pursuit_Cost','Pursuit_Cost_FY24_26','Pursuit_Cost_last_month']):
        # safe guards for missing KPI columns
        cols = [c for c in kpis if c in subdf.columns]
        if len(cols) == 0:
            return pd.DataFrame({"note": ["no kpi cols present"]})
        na_0 = [int(subdf[c].isna().sum() + subdf[subdf[c]==0].shape[0]) for c in cols]
        total = [float(subdf[c].sum()) for c in cols]
        df_summary_sum = pd.DataFrame(data = {'total': total, 'na_or_zero': na_0}, index=cols)
        return df_summary_sum

    # define row masks
    masks = [
        ("High >0.85", df[model_pred_col] > 0.85),
        ("Low <0.15", df[model_pred_col] < 0.15),
        ("Medium 0.15-0.85", (df[model_pred_col] >= 0.15) & (df[model_pred_col] <= 0.85))
    ]

    # make column 3 (KPI) slightly narrower so tables don't overlap CM
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1.0, 1.0, 0.9], height_ratios=[2, 2, 2])

    for row_idx, (row_label, mask) in enumerate(masks):
        subdf = df[mask].copy()
        # compute cms and acc
        cm_model, cm_deal, acc_model, acc_deal = _cm_and_acc(subdf)

        # Model confusion matrix (col 0)
        ax_m = fig.add_subplot(gs[row_idx, 0])
        if cm_model is not None:
            title_m = f"{row_label} - Model (n={len(subdf)})\nAcc={acc_model:.2f}"
            plot_confusion_matrix_only(cm_model, prefix=title_m, ax=ax_m)
        else:
            ax_m.axis('off')
            ax_m.text(0.5, 0.5, "No data", ha='center', va='center')

        # Deal confusion matrix (col 1)
        ax_d = fig.add_subplot(gs[row_idx, 1])
        if cm_deal is not None:
            title_d = f"{row_label} - Deal (n={len(subdf)})\nAcc={acc_deal:.2f}"
            plot_confusion_matrix_only(cm_deal, prefix=title_d, ax=ax_d)
        else:
            ax_d.axis('off')
            ax_d.text(0.5, 0.5, "Deal probabilities\nnot available", ha='center', va='center')

        # KPI summary (col 2)
        ax_k = fig.add_subplot(gs[row_idx, 2])
        df_kpi = generate_kpi_sum(subdf)
        ax_k.axis('off')
        if df_kpi.shape[0] == 1 and 'note' in df_kpi.columns:
            ax_k.text(0.5, 0.5, df_kpi.iloc[0,0], ha='center', va='center')
        else:
            # render table with smaller font and scaled cells to avoid overlap
            cell_text = df_kpi.round(2).astype(object).values
            tbl = ax_k.table(cellText=cell_text,
                             rowLabels=df_kpi.index.astype(str),
                             colLabels=df_kpi.columns.astype(str),
                             loc='center', cellLoc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.2)
            # smaller title to save vertical space
            ax_k.set_title(f"{row_label} - KPI Summary (n={len(subdf)})", fontsize=10, pad=6)

    # tighten spacing and ensure enough horizontal gap to prevent overlap
    plt.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.03, wspace=0.46, hspace=0.36)

    #plt.suptitle(f"Model vs Deal comparison {prefix}".strip(), fontsize=14)
    save_path = f"data/titanic_survived/{prefix}_cm_pred_vs_deal.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def log_feature_importance(model, feature_names, mlflow, n=10, filename="feature_importance.png"):
    """Plot and log top n feature importances for a model."""
    #import matplotlib.pyplot as plt
    #import numpy as np
    feature_importances = model.feature_importances_
    n = min(n, len(feature_importances))
    # Get indices of top n features
    top_idx = np.argsort(feature_importances)[-n:]
    top_features = np.array(feature_names)[top_idx]
    top_importances = feature_importances[top_idx]
    # Sort for better visualization
    sorted_idx = np.argsort(top_importances)
    top_features = top_features[sorted_idx]
    top_importances = top_importances[sorted_idx]
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_importances)
    plt.xlabel("Feature Importance")
    plt.title(f"Top {n} Feature Importances")
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / filename
    plt.savefig(plot_path)
    mlflow.log_artifact(str(plot_path))
    plt.close()
    return top_features, top_importances

def log_evaluation_and_artifacts(model, X, y, prefix, mlflow, pursuit_df=None, extra_merge_col="Opportunity_Code", deal_prob_col="Deal_Win_Probab"):
    metrics, cm, pred = evaluate_model(model, X, y, prefix=prefix)
    for k, v in metrics.items():
        if k == f"{prefix}accuracy":
            mlflow.log_metric(k, v)
    roc_path = plot_roc(y, pred["Predicted_Probab"], prefix=prefix.rstrip('_'))
    cm_path = plot_confusion_matrix(cm, pred["Actual"], pred["Prediction"], prefix=prefix.rstrip('_'))
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(cm_path)
    # scatter plot of Predicted_Probab vs Actual with color showing Actual and title showing prefix
    if ROW_ID_COL not in pred.columns:
        pred = pred.reset_index().rename(columns={"index": ROW_ID_COL})
    df_scatter = pred[[ROW_ID_COL, "Predicted_Probab", "Actual"]].copy()

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(data=df_scatter, x="Predicted_Probab", y=ROW_ID_COL, c=df_scatter["Actual"], cmap='bwr', alpha=0.7)
    plt.title(f"Titanic Survived Scatter Plot ({prefix.rstrip('_')})")
    plt.xlabel("Titanic Survived Probability")
    plt.ylabel("Row")
    plt.legend(title="Survived")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"{prefix}_titanic_survived_scatter.png"
    plt.savefig(plot_path)
    plt.close(fig)
    mlflow.log_artifact(str(plot_path))
    return pred

def melt_shap_top_features(shap_df, id_col="Name", top_n=10):
    """
    Melts a SHAP value DataFrame, computes direction, absolute value, proportion, and feature importance,
    and returns the top N features per record.

    Parameters:
        shap_df (pd.DataFrame): SHAP values DataFrame with id_col and feature columns.
        id_col (str): Name of the ID column
        top_n (int): Number of top features to keep per record.

    Returns:
        pd.DataFrame: Melted DataFrame with top N features per record.
    """
    shap_melted = shap_df.melt(id_vars=id_col, var_name="Feature", value_name="SHAP_Value")
    shap_melted["Direction"] = shap_melted["SHAP_Value"].apply(lambda x: 1 if x >= 0 else -1)
    shap_melted["Abs_SHAP_Value"] = shap_melted["SHAP_Value"].abs()
    shap_melted["Proportion"] = shap_melted["Abs_SHAP_Value"] * 100 / shap_melted.groupby(id_col)["Abs_SHAP_Value"].transform("sum")
    shap_melted["Proportion"] = shap_melted["Proportion"].round(1)
    shap_melted = shap_melted.sort_values([id_col, "Abs_SHAP_Value"], ascending=[True, False])
    shap_melted = shap_melted.groupby(id_col).head(top_n)
    shap_melted["Feature_Importance"] = shap_melted["Direction"] * shap_melted["Proportion"]
    # round Feature_Importance to 1 decimal place
    shap_melted["Feature_Importance"] = shap_melted["Feature_Importance"].round(1)
    # filter columns Opportunity_Code, Feature,Feature_Importance
    shap_melted = shap_melted[[id_col, "Feature", "Feature_Importance"]]
    return shap_melted

def generate_shap_results(model, x_data, prediction_data, prefix, config, mlflow) -> pd.DataFrame:
    """
    Generate SHAP values, summary plot, and formatted results for validation data.
    Args:
        model: Trained model object
        x_data: input featutres (DataFrame)
        prediction_data: DataFrame with validation predictions (must include Opportunity_Code)
        prefix: String prefix for output files (e.g., 'val')
        config: Configuration dictionary (must include output_dir)
    Returns:
        DataFrame with SHAP top features merged with predictions
    """
    explainer = shap.Explainer(model)
    shap_values_val = explainer(x_data)
    # plot the shap summary plot
    plt.figure(figsize=(12, 8))  # Increase width as needed
    shap.summary_plot(shap_values_val, x_data, show=False, max_display=20)
    plt.tight_layout()
    shap_summary_path = f"{config['data']['output_dir']}/shap_{prefix}_plot.png"
    plt.savefig(shap_summary_path)
    plt.close()
    mlflow.log_artifact(shap_summary_path)
    # Optionally log artifact if mlflow is available in the calling scope
    # get contribution of each feature to the prediction for each record in X_val
    shap_df_val = pd.DataFrame(shap_values_val.values, columns=x_data.columns, index=x_data.index)
    shap_df_val = shap_df_val.reset_index().rename(columns={"index": ROW_ID_COL})
    shap_melted_val = melt_shap_top_features(shap_df_val, id_col='Name')
    # take Feature and Feature_Importance columns and convert into a comma separated string for each Opportunity_Code 
    shap_melted_val = shap_melted_val.groupby('Name').apply(lambda x: ', '.join(x["Feature"] + " (" + x["Feature_Importance"].astype(str) + ")")).reset_index(name="Top_Features")
    # merge predictions with top features by row id
    shap_melted_val = pd.merge(prediction_data, shap_melted_val, on='Name', how="left")
    # rename Survived to Actual_Survived if present
    if "Survived" in shap_melted_val.columns:
        shap_melted_val = shap_melted_val.rename(columns={"Survived": "Actual_Survived"})
    # move Actual Survived column to specific position
    first_6_cols = ['Name', "Actual", "Prediction", "Predicted_Probab", "Titanic_Survived", "Top_Features"]
    shap_melted_val = shap_melted_val[first_6_cols+[col for col in shap_melted_val.columns if col not in first_6_cols]]
    return shap_melted_val

def main(config: Dict[str, Any]):
    mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_modeling")
    # get run name as input from user
    run_name = input("Enter run name: ")
    with mlflow.start_run(run_name=run_name) as mlflow_run:
        output_dir = Path(config["data"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        # Load data
        train_df, test_df = load_data(config)
        X_train, y_train = split_features_target(train_df, config["modeling"]["target"])
        X_test, y_test = split_features_target(test_df, config["modeling"]["target"])

        if config["modeling"].get("use_gridsearch", False):
            grid_search = train_model_with_gridsearch(X_train, y_train, config)
            model = grid_search.best_estimator_     # get best model from grid search
            # Log grid search results and heatmap
            log_gridsearch_results(grid_search, mlflow)
            heatmap_path = plot_gridsearch_heatmap(grid_search)
            mlflow.log_artifact(heatmap_path)
        else:
            model = train_model(X_train, y_train, config)

        # get model accuracy on training set
        train_metrics, _, _ = evaluate_model(model, X_train, y_train, prefix="train_")
        mlflow.log_metric("train_accuracy", train_metrics["train_accuracy"])
        input_example = X_train.head(1)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="titanic_survived_model_sklearn",
            input_example=input_example,
            signature=signature
        )
        # log feature importance plot 
        top_feat, feat_score = log_feature_importance(model, X_train.columns, mlflow, n=10)
        # invert order so the best (highest importance) appears first when the function returned ascending order
        top_feat = list(top_feat)[::-1]
        feat_score = list(feat_score)[::-1]
        # create somma seperated string showing feature name and importance 
        model_feat_imp = ", ".join([f"{f} : {s:.2f}" for f, s in zip(top_feat, feat_score)])
        # Evaluate and log for test set
        test_pred = log_evaluation_and_artifacts(model, 
                                                 X_test, 
                                                 y_test, 
                                                 "test_", 
                                                 mlflow)

        # generate SHAP results for validation data
        df_shap_test = generate_shap_results(model, X_test, test_pred, 'test', config, mlflow)
        df_shap_test['Model_Feat_Imp'] = model_feat_imp

        predictions_path = output_dir / "predictions_test.xlsx"
        shap_path = output_dir / "shap_predictions_test.xlsx"
        test_pred.to_excel(predictions_path, sheet_name="Test_Predictions", index=False)
        df_shap_test.to_excel(shap_path, sheet_name="Test_Predictions", index=False)
        print("Model training and evaluation complete. Metrics and plots logged to MLflow.")

if __name__ == "__main__":
    with open("poc/src/classification/config.yaml", "r") as f:
        config = yaml.safe_load(f) 
    main(config)