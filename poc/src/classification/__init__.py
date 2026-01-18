"""Classification package.

Public API:
- Config: pydantic model with pipeline settings
- load_data, preprocess, split: data helpers
- train_model: model training function
- evaluate: evaluation function
- run: top-level pipeline runner
"""

from .config import Config
from .data import load_data, preprocess, split
from .model import train_model
from .evaluate import evaluate
from .pipeline import run

__all__ = [
    "Config",
    "load_data",
    "preprocess",
    "split",
    "train_model",
    "evaluate",
    "run",
]

__version__ = "0.1.0"
