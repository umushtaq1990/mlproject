from poc.src.classification.config import Config
from poc.src.classification.data import load_data, preprocess, split
from poc.src.classification.model import train_model
from poc.src.classification.evaluate import evaluate


def run():
    cfg = Config()
    df = load_data(cfg.data_path)
    X, y = preprocess(df, cfg.target_col)
    X_train, X_test, y_train, y_test = split(X, y, cfg.test_size, cfg.random_state)

    model = train_model(X_train, y_train)
    acc = evaluate(model, X_test, y_test)
    print(f"Accuracy: {acc:.2f}")


if __name__ == "__main__":
    run()
