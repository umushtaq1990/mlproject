import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from poc.src.classification.config import FeatureConfig, NumericFeatureConfig, CategoricalFeatureConfig

# -------------------------------
# Numeric Feature Functions
# -------------------------------
def handle_numeric_nans(df: pd.DataFrame, col: str, cfg: NumericFeatureConfig) -> pd.Series:
    """Fill missing numeric values based on the strategy defined in config."""
    if cfg.handle_nans == "mean":
        return df[col].fillna(df[col].mean())
    elif cfg.handle_nans == "median":
        return df[col].fillna(df[col].median())
    elif cfg.handle_nans == "constant":
        return df[col].fillna(cfg.fill_value)
    return df[col]


def remove_outliers(df: pd.DataFrame, col: str, cfg: NumericFeatureConfig) -> pd.DataFrame:
    """Remove outliers using the IQR method if enabled in config."""
    if not cfg.outlier_treatment or col not in df:
        return df
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]


def scale_numeric(df: pd.DataFrame, col: str, cfg: NumericFeatureConfig) -> pd.Series:
    """Scale numeric column using the strategy defined in config."""
    if cfg.scaling == "standard":
        return pd.Series(StandardScaler().fit_transform(df[[col]]).flatten(), name=col)
    elif cfg.scaling == "minmax":
        return pd.Series(MinMaxScaler().fit_transform(df[[col]]).flatten(), name=col)
    elif cfg.scaling == "robust":
        return pd.Series(RobustScaler().fit_transform(df[[col]]).flatten(), name=col)
    return df[col]


def plot_numeric(df: pd.DataFrame, col: str, stage: str = "raw", output_dir: str = "plots"):
    """
    Plots numeric feature distribution.
    
    Args:
        df: DataFrame containing the column.
        col: Column name to plot.
        stage: "raw" or "processed" (used in filename and title)
        output_dir: Directory to save plots.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"{stage.capitalize()} Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{col}_{stage}_distribution.png")
    plt.close()


# -------------------------------
# Categorical Feature Functions
# -------------------------------
def handle_categorical_nans(df: pd.DataFrame, col: str, cfg: CategoricalFeatureConfig) -> pd.Series:
    """Fill missing categorical values based on strategy in config."""
    if cfg.handle_nans == "mode":
        return df[col].fillna(df[col].mode()[0])
    elif cfg.handle_nans == "constant":
        return df[col].fillna(cfg.fill_value)
    return df[col]


def plot_categorical_frequency(df: pd.DataFrame, col: str, stage: str = "raw", output_dir: str = "plots"):
    """
    Plots categorical feature frequency.
    
    Args:
        df: DataFrame containing the column.
        col: Column name to plot.
        stage: "raw" or "processed"
        output_dir: Directory to save plots.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(6, 4))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"{stage.capitalize()} Frequency of {col}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{col}_{stage}_frequency.png")
    plt.close()


# -------------------------------
# NaN Summary Plot
# -------------------------------
def plot_nan_summary(df: pd.DataFrame, output_dir: str = "plots"):
    """Plot number of NaNs per feature."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    nan_counts = df.isna().sum()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=nan_counts.index, y=nan_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of NaNs")
    plt.title("NaN Summary per Feature")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/nan_summary.png")
    plt.close()


# -------------------------------
# Main Feature Processing Function
# -------------------------------
def feature_engineering(df: pd.DataFrame, feature_cfg: FeatureConfig) -> pd.DataFrame:
    """
    Process numeric and categorical features according to the feature config.
    Generates initial NaN plots and raw numeric distributions.
    Then applies NaN handling, outlier removal, scaling, and encoding.
    """
    df_processed = df.copy()

    # -------------------------------
    # 1️⃣ Generate EDA plots before processing
    # -------------------------------
    plot_nan_summary(df_processed)

    numeric_cols = list(feature_cfg.numeric_features.keys())
    for col in numeric_cols:
        if col in df_processed.columns:
            plot_numeric(df_processed, col, stage="raw")

    categorical_cols = list(feature_cfg.categorical_features.keys())
    for col in categorical_cols:
        if col in df_processed.columns:
            plot_categorical_frequency(df_processed, col, stage="raw")

    # -------------------------------
    # 2️⃣ Process Numeric Features
    # -------------------------------
    for col, cfg in feature_cfg.numeric_features.items():
        if col not in df_processed.columns:
            continue
        df_processed[col] = handle_numeric_nans(df_processed, col, cfg)
        df_processed = remove_outliers(df_processed, col, cfg)
        df_processed[col] = scale_numeric(df_processed, col, cfg)
        plot_numeric(df_processed, col, stage="processed")

    # -------------------------------
    # 3️⃣ Process Categorical Features
    # -------------------------------
    for col, cfg in feature_cfg.categorical_features.items():
        if col not in df_processed.columns:
            continue
        df_processed[col] = handle_categorical_nans(df_processed, col, cfg)
        plot_categorical_frequency(df_processed, col, stage="processed")

    return df_processed
