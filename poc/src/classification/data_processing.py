from sklearn.cluster import KMeans
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from poc.src.classification.config import load_config, load_feature_config
from poc.src.classification.feature_processing import feature_engineering

#from utils import v

import pandas as pd
from pathlib import Path
from typing import Any

def process_data(df: pd.DataFrame, config: Any) -> pd.DataFrame:
    """
    Cleans and processes the raw DataFrame according to the config.
    Returns a DataFrame with desired data types and cleaned values.
    Automatically handles:
      - column selection
      - renaming
      - dropping
      - numeric/categorical column handling
      - saving processed data
    """
    # 1️⃣ Select only the columns defined in config
    df_f = df[config.processing.raw_columns].copy()

    # 2️⃣ Rename columns
    if config.processing.rename_columns:
        df_f = df_f.rename(columns=config.processing.rename_columns)

    # 3️⃣ Drop unwanted columns if they exist
    if config.processing.drop_columns:
        df_f = df_f.drop(columns=config.processing.drop_columns, errors="ignore")

    # 4️⃣ Optional: cast numeric and categorical columns
    for col in config.processing.numeric_cols:
        if col in df_f.columns:
            df_f[col] = pd.to_numeric(df_f[col], errors="coerce")

    for col in config.processing.categorical_cols:
        if col in df_f.columns:
            df_f[col] = df_f[col].astype("category")

    # 5️⃣ Ensure processed output folder exists
    processed_path = Path(config.data.processed)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # 6️⃣ Save processed data
    df_f.to_parquet(processed_path, index=False)

    return df_f

def bin_activeness_score(scores, method='kmeans', n_bins=4, labels=None):
    """
    Bin activeness scores into categories using kmeans or qcut.
    method: 'kmeans' or 'qcut'
    n_bins: number of bins (e.g., 4 for Low/Medium/High/VeryHigh)
    labels: list of labels for bins
    Returns: list of bin labels
    """
    if labels is None:
        if n_bins == 4:
            labels = ['Low', 'Medium', 'High', 'VeryHigh']
        elif n_bins == 3:
            labels = ['Low', 'Medium', 'High']
        else:
            labels = [f'Bin{i+1}' for i in range(n_bins)]
    scores_arr = np.array(scores).reshape(-1, 1)
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_bins, random_state=0, n_init=10).fit(scores_arr)
        bins = kmeans.predict(scores_arr)
        centers = kmeans.cluster_centers_.flatten()
        order = np.argsort(centers)
        label_map = {order[i]: labels[i] for i in range(n_bins)}
        return [label_map[b] for b in bins]
    elif method == 'qcut':
        try:
            quantile_scores = pd.qcut(scores, q=n_bins, labels=labels)
        except ValueError as e:
            logging.warning(f"Error in qcut {n_bins} binning: {e} \n trying with n_bins-1")
            try:
                quantile_scores = pd.qcut(scores, q=n_bins-1, labels=labels)
            except ValueError as e:
                logging.warning(f"Error in qcut {n_bins-1} binning: {e} \n trying with n_bins-2")
                try:
                    quantile_scores = pd.qcut(scores, q=n_bins-2, labels=labels)
                except ValueError as e:
                    logging.warning(f"Error in qcut {n_bins-2} binning: {e} \n giving up")
                    quantile_scores = pd.Series([labels[0]] * len(scores))
        return quantile_scores
    else:
        raise ValueError('Unknown binning method')

# === Helper Functions for Activeness Category Generation ===
def cap_outliers(series, quantile=0.99):
    upper_limit = series.quantile(quantile)
    return series.clip(upper=upper_limit)

def minmax_scale(series):
    min_score = series.min()
    max_score = series.max()
    if max_score - min_score == 0:
        return series * 0  # all zeros if constant
    return (series - min_score) / (max_score - min_score)

def generate_binnings(series: pd.Series, n_bins: int, labels: list[str]) -> Tuple[List[str], List[str]]:
    kmeans_labels = bin_activeness_score(series, method='kmeans', n_bins=n_bins, labels=labels)
    qcut_labels = bin_activeness_score(series, method='qcut', n_bins=n_bins, labels=labels)
    return kmeans_labels, qcut_labels

def plot_activeness_bins(df:pd.DataFrame, value_col:str, kmeans_col:str, qcut_col:str, dir_path:str):
    plot_dir = Path(dir_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))
    # Violin plot for KMeans bins
    plt.subplot(1, 2, 1)
    sns.violinplot(x=kmeans_col, y=value_col, data=df, inner='box', palette='Set2')
    sns.stripplot(x=kmeans_col, y=value_col, data=df, color='k', size=2, jitter=True, alpha=0.5)
    plt.title('Activeness Score by KMeans Bins')
    plt.xlabel('KMeans Bins')
    plt.ylabel('Activeness Score')
    # Violin plot for Quartile bins
    plt.subplot(1, 2, 2)
    sns.violinplot(x=qcut_col, y=value_col, data=df, inner='box', palette='Set3')
    sns.stripplot(x=qcut_col, y=value_col, data=df, color='k', size=2, jitter=True, alpha=0.5)
    plt.title('Activeness Score by Quartile Bins')
    plt.xlabel('Quartile Bins')
    plt.ylabel('Activeness Score')
    plt.tight_layout()
    plt.show()
    plot_path = plot_dir / f"{value_col}.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved activeness bins plot to {plot_path}")

def compare_binnings(df:pd.DataFrame, 
                     numeric_col:str, 
                     group_col:Dict[str, Any],
                     dir_path:str):
    logging.info(f"Generating and comparing binnings for column: {numeric_col}")
    if group_col.outlier_treatment:
        df[numeric_col] = cap_outliers(df[numeric_col])
    if group_col.scaling!='none':
        df[numeric_col] = minmax_scale(df[numeric_col])
    kmeans_labels, qcut_labels = generate_binnings(df[numeric_col], n_bins=group_col.n_bins, labels=group_col.labels)
    df[f'{numeric_col}_Active_kmean'] = kmeans_labels
    df[f'{numeric_col}_Active_quartile'] = qcut_labels
    plot_activeness_bins(df, 
                         numeric_col, 
                         f'{numeric_col}_Active_kmean', 
                         f'{numeric_col}_Active_quartile',
                         dir_path=dir_path)
    # get back kmeans labels column only 
    df[numeric_col] = df[f'{numeric_col}_Active_kmean']
    df = df.drop(columns=[f'{numeric_col}_Active_kmean', f'{numeric_col}_Active_quartile'])
    return df

def generate_activeness_category_wrt_numeric_col(
    df: pd.DataFrame, 
    group_col: list[str], 
    numeric_col: Dict[str, Any],
    config: Dict[str, Any],
):
    """
    Generates a categorical activeness feature for combinations of groupby_cols with respect to target_col and experience volume.
    Merges the result back to the input DataFrame and returns it.
    """
    groupby_cols = group_col.columns
    score_col = '_'.join(groupby_cols) + f'_ActiveScore_{numeric_col.column}'
    new_feature_name = '_'.join(groupby_cols) + f'_ActiveBin_{numeric_col.column}'
    # log message
    logging.info(f"Generating activeness category with respect to {numeric_col} for groups: {groupby_cols}")
    survived_df = df[df[config.modeling.target] == config.modeling.positive_label]
    # Group and sum # TODO log message
    df_sum = survived_df.groupby(groupby_cols)[numeric_col.column].sum().reset_index()
    lost_df = df[df[config.modeling.target] == config.modeling.negative_label]
    df_sum_lost = lost_df.groupby(groupby_cols)[numeric_col.column].sum().reset_index()
    df_sum_survived_lost = pd.merge(df_sum, df_sum_lost, on=groupby_cols, how='left', suffixes=('_survived', '_lost'))
    # fill na with 0
    df_sum_survived_lost[f'{numeric_col.column}_survived'] = df_sum_survived_lost[f'{numeric_col.column}_survived'].fillna(0)
    df_sum_survived_lost[f'{numeric_col.column}_lost'] = df_sum_survived_lost[f'{numeric_col.column}_lost'].fillna(0)
    df_sum_survived_lost[numeric_col.column] = df_sum_survived_lost[f'{numeric_col.column}_survived'].fillna(0) + df_sum_survived_lost[f'{numeric_col.column}_lost'].fillna(0)
    # get score as survived / (survived + lost)
    epsilon = 1e-6
    df_sum_survived_lost[numeric_col.column] = df_sum_survived_lost[f'{numeric_col.column}_survived'] / (df_sum_survived_lost[f'{numeric_col.column}_survived'] + df_sum_survived_lost[f'{numeric_col.column}_lost'] + epsilon)
    df_sum = df_sum_survived_lost[groupby_cols + [numeric_col.column]]
    score_col = f"{score_col}_Conv"
    new_feature_name = f"{new_feature_name}_Conv"
    df_sum[score_col] = df_sum[numeric_col.column]
    df_sum[new_feature_name] = df_sum[numeric_col.column]
    df_sum = compare_binnings(df_sum, new_feature_name, group_col=group_col, dir_path="plots")
    df_grouped_cluster = df_sum[groupby_cols + [score_col, new_feature_name]]
    df_grouped_cluster.to_excel(f"{config.data.output_dir}/{new_feature_name}.xlsx", index=False)
    df_out = pd.merge(df, df_grouped_cluster, on=groupby_cols, how='left')
    return df_out, new_feature_name

def generate_activeness_category(
    df: pd.DataFrame, 
    group_col: Dict[str, Any], 
    config: Dict[str, Any],
):
    """
    Generates a categorical activeness feature for combinations of groupby_cols with respect to target_col.
    Merges the result back to the input DataFrame and returns it.
    """
    groupby_cols = group_col.columns
    score_col = '_'.join(groupby_cols) + f'_ActiveScore_{config.modeling.target}'
    new_feature_name = '_'.join(groupby_cols) + f'_ActiveBin_{config.modeling.target}'
    logging.info(f"Generating activeness category with respect to {config.modeling.target} for groups: {groupby_cols}")
    # filter df to only positive and negative labels
    df_closed = df[df[config.modeling.target].isin([config.modeling.positive_label, config.modeling.negative_label])]
    # Group and count
    df_score = (
        df_closed.groupby(groupby_cols + [config.modeling.target])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    for col in [config.modeling.positive_label, config.modeling.negative_label]:
        if col not in df_score.columns:
            df_score[col] = 0
    epsilon = 1e-6
    
    score_col = f"{score_col}_Conv"
    df_score[score_col] =  df_score[config.modeling.positive_label] / (df_score[config.modeling.positive_label] + df_score[config.modeling.negative_label] + epsilon)
    new_feature_name = f"{new_feature_name}_Conv"

    df_score[new_feature_name] = df_score[score_col]
    df_score = compare_binnings(df_score, new_feature_name, group_col=group_col, dir_path="plots")
    df_grouped_cluster = df_score[groupby_cols + [score_col, new_feature_name]]
    df_grouped_cluster.to_parquet(f"{config.data.output_dir}/{new_feature_name}.parquet", index=False)
    #df_out = pd.merge(df, df_grouped_cluster, on=groupby_cols, how='left')
    # remove cases where merge results in nan (due to no closed won/lost deals for that group)
    df_out = pd.merge(df, df_grouped_cluster, on=groupby_cols)
    return df_out, new_feature_name

def feature_analysis(df, config):
    feat_config = load_feature_config(config.data.feat_config_path)
    df_processed = feature_engineering(df, feat_config)
    grouped_cols= []
    for group_col in feat_config.conversion_features.groupby_combinations: #different combinations could be tried if more categorical variables are available
        df_processed, col_name = generate_activeness_category(
            df_processed,
            group_col=group_col,
            config=config
        )
        grouped_cols.append(col_name)
        # generate activeness category wrt numeric columns present in conversion_numeric_cols for each group_col
        for numeric_col in feat_config.conversion_features.numeric_col:
            df_processed, col_name_age = generate_activeness_category_wrt_numeric_col(   
                df_processed,
                group_col=group_col,
                numeric_col=numeric_col,
                config=config
            )
            grouped_cols.append(col_name_age)
    return df_processed, grouped_cols

def split_train_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_kwargs = {
        "test_size": test_size,
        "random_state": random_state,
        "shuffle": shuffle,
    }
    if target_col in df.columns:
        split_kwargs["stratify"] = df[target_col]
    return train_test_split(df, **split_kwargs)

def save_train_test_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: Path,
    test_path: Path,
) -> None:
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=True)
    test_df.to_parquet(test_path, index=True)

def prepare_model_data(df_model, config):
    #df_model['Age'] = bin_activeness_score(df_model['Age'], 
    #                                       method='kmeans', 
    #                                       n_bins=4, 
    #                                       labels=['Young', 'MidAge', 'Senior', 'Veteran'])
    activeness_score_cols = [col for col in df_model.columns if 'ActiveScore' in col]
    check_activeness_cols = [col for col in activeness_score_cols if col in df_model.columns]
    df_model = df_model.drop(columns=check_activeness_cols)
    df_model = df_model.set_index(["Name"])
    df_model = pd.get_dummies(df_model, drop_first=True)
    # make sure to convert all bool columns to int
    bool_cols = df_model.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_model[col] = df_model[col].astype(int)
    # remove rows with any remaining NaN values
    df_model = df_model.dropna()
    train_df, test_df = split_train_test(
        df_model,
        target_col=config.modeling.target,
        test_size=config.processing.test_size,
        random_state=config.processing.random_state,
        shuffle=True,
    )

    save_train_test_splits(
        train_df,
        test_df,
        train_path=Path(config.data.train),
        test_path=Path(config.data.test),
    )
    return df_model

# =========================
# Logging Setup
# =========================

def setup_logger(name: str = "win_prob_logger", log_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

# =========================
# Data Loader
# =========================

class DataLoader:
    """Load raw data."""
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            df_raw = pd.read_csv(self.config.data.raw)
            self.logger.info(f"Loaded raw data shape: {df_raw.shape}")
            return df_raw
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

# =========================
# Feature Engineering (use your existing functions)
# =========================

class FeatureEngineer:
    """Handles all feature engineering steps."""
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Use your existing feature engineering pipeline here
        self.logger.info("Starting feature engineering pipeline.")
        # process deal dataframe with selected columns
        df_processed = process_data(df, config=self.config)
        df_final_analysis, client_activness_cols = feature_analysis(df_processed, self.config)
        return df_final_analysis, client_activness_cols

# =========================
# Data Preparation
# =========================

class DataPreparer:
    """Prepares data for ML modeling: cleaning, encoding, splitting."""
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Preparing data for ML modeling.")
        df_model = prepare_model_data(df, self.config)
        return df_model

# =========================
# Pipeline Orchestration
# =========================

class MainPipeline:
    """Orchestrates the full win probability data processing pipeline."""
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = setup_logger()

    def run_pipeline(self):
        self.logger.info("Pipeline started.")
        mlflow.set_experiment(f"{self.config.mlflow.experiment_name}_data_processing")
        # get run name as input from user
        run_name = input("Enter run name: ")
        mlflow.start_run(run_name=run_name)
        try:
            # Data Loading
            data_loader = DataLoader(self.config, self.logger)
            df_raw = data_loader.load()
            mlflow.log_metrics({
                "raw_rows": df_raw.shape[0],
                "raw_columns": df_raw.shape[1],
            })
            # Feature Engineering
            feature_engineer = FeatureEngineer(self.config, self.logger)
            df_features, client_activness_cols = feature_engineer.process(df_raw)
            mlflow.log_metric("processed_data_rows", df_features.shape[0])

            # Data Preparation
            data_preparer = DataPreparer(self.config, self.logger)
            df_model = data_preparer.prepare(df_features)
            mlflow.log_metric("final data rows", df_model.shape[0])
            mlflow.log_param("final data columns", df_model.shape[1])

            self.logger.info("Pipeline completed successfully.")
            mlflow.log_param("status", "success")
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            mlflow.log_param("status", "failed")
            raise
        finally:
            mlflow.end_run()

# =========================
# Entry Point
# =========================

def main():
    """
    Entry point for the win probability pipeline.
    """
    config_path = "poc/src/classification/config.yaml"
    pipeline = MainPipeline(config_path)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
