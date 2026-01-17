from pydantic import BaseModel
from pathlib import Path


class Config(BaseModel):
    data_path: Path = Path("poc/data/dummy.csv")
    target_col: str = "Survived"
    test_size: float = 0.2
    random_state: int = 42