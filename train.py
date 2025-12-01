# train.py
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

PREPROCESSED_PATH = "data/dataset_preprocessed.csv"

def get_model(name):
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if name == "svm":
        return SVC(kernel="rbf", probability=True)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    raise ValueError(f"Unknown model: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rf",
                        help="Choose: rf, svm, knn")
    args = parser.parse_args()

    df = pd.read_csv(PREPROCESSED_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = get_model(args.model)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    # Save metrics
    metrics_path = f"metrics_{args.model}.txt"
    with open(metrics_path, "w") as f:
        f.write(f"MODEL: {args.model}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    joblib.dump(model, f"model_{args.model}.joblib")

    print(f"Saved {metrics_path} and model_{args.model}.joblib")

if __name__ == "__main__":
    main()
