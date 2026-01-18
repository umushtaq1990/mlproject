import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)


def preprocess(df, target):
    df = df.copy()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
