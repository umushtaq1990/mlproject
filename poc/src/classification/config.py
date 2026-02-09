from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


# -------------------------
# Section-level schemas
# -------------------------
class DataConfig(BaseModel):
    raw_path: str
    train_path: str
    test_path: str
    validation_path: str
    processed_path: str
    output_dir: Path
    feat_config_path: str

    @property
    def raw(self) -> Path:
        return Path(self.raw_path.format(output_dir=self.output_dir))
    @property
    def train(self) -> Path:
        return Path(self.train_path.format(output_dir=self.output_dir))
    @property
    def test(self) -> Path:
        return Path(self.test_path.format(output_dir=self.output_dir))
    @property
    def validation(self) -> Path:
        return Path(self.validation_path.format(output_dir=self.output_dir))
    @property
    def processed(self) -> Path:
        return Path(self.processed_path.format(output_dir=self.output_dir))

class ProcessingConfig(BaseModel):
    raw_columns: List[str]
    rename_columns: Dict[str, str] = Field(default_factory=dict)
    drop_columns: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]

    test_size: float = Field(gt=0, lt=1)
    random_state: int
class ModelParams(BaseModel):
    params: Dict[str, Any] = {}
    hyperparameters: Dict[str, Any] = {}


class ModelingConfig(BaseModel):
    use_gridsearch: bool
    target: str
    positive_label:int
    negative_label:int
    model: ModelParams


class MlflowConfig(BaseModel):
    experiment_name: str


class NumericFeatureConfig(BaseModel):
    handle_nans: Literal["mean", "median", "constant"]
    fill_value: Optional[float] = None
    outlier_treatment: bool = False
    scaling: Literal["standard", "minmax", "robust", "none"] = "none"

class CategoricalFeatureConfig(BaseModel):
    handle_nans: Literal["mode", "constant"]
    fill_value: Optional[str] = None
    encoding: Literal["onehot", "label", "none"] = "none"

class BinningConfig(BaseModel):
    columns: List[str]
    n_bins: int
    labels: List[str] = Field(default_factory=list)
    outlier_treatment: bool = False
    scaling: Literal["standard", "minmax", "robust", "none"] = "none"

class NumericBinningConfig(BaseModel):
    column: str
    n_bins: int
    labels: List[str] = Field(default_factory=list)
    outlier_treatment: bool = False
    scaling: Literal["standard", "minmax", "robust", "none"] = "none"

class ConversionFeaturesConfig(BaseModel):
    groupby_combinations: List[BinningConfig] = Field(default_factory=list)
    numeric_col: List[NumericBinningConfig] = Field(default_factory=list)

class FeatureConfig(BaseModel):
    numeric_features: Dict[str, NumericFeatureConfig] = Field(default_factory=dict)
    categorical_features: Dict[str, CategoricalFeatureConfig] = Field(default_factory=dict)
    conversion_features: ConversionFeaturesConfig = Field(default_factory=ConversionFeaturesConfig)

# -------------------------
# Root config (ties all together)
# -------------------------

class AppConfig(BaseModel):
    data: DataConfig
    processing: ProcessingConfig
    modeling: ModelingConfig
    mlflow: MlflowConfig


# -------------------------
# Loader function (single entry point)
# -------------------------

def load_config(config_path: str | Path) -> AppConfig:
    """
    Load a YAML config file and validate it using Pydantic.

    Fails fast if config is invalid.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    try:
        return AppConfig.model_validate(raw_config)
    except ValidationError as e:
        raise RuntimeError(
            f"❌ Invalid configuration file: {path}\n{e}"
        ) from e

def load_feature_config(config_path: str | Path) -> FeatureConfig:
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f) or {}
    return FeatureConfig(**raw_config)