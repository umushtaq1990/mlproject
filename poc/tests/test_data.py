from poc.src.classification.data import preprocess

import pandas as pd


def test_preprocess():
    df = pd.DataFrame({"Sex": ["male", "female"], "Survived": [0, 1]})

    X, y = preprocess(df, "Survived")
    assert "Sex" in X.columns
    assert y.sum() == 1
