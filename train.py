import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_config():
    """Load the YAML configuration file."""
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)


def build_model(model_type, params):
    """Create a sklearn model based on type + parameters."""
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    elif model_type == "decision_tree":
        return DecisionTreeClassifier(**params)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test):
    """Train a model, evaluate it, and save metrics to a file."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Save metrics for this model
    metrics_file = f"metrics_{model_name}.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")

    print(f"✔ Saved metrics for {model_name} → {metrics_file} (accuracy={acc:.4f})")


def main():
    # Load config
    config = load_config()
    print

    # Load dataset
    df = pd.read_csv(config["dataset"]["path"])

    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # Build and train models from config
    for model_name, model_cfg in config["models"].items():
        print(f"\n--- Training {model_name} ---")
        model = build_model(model_cfg["type"], model_cfg["params"])
        train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
