# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

PREPROCESSED_PATH = "data/dataset_preprocessed.csv"
METRICS_PATH = "metrics.txt"
MODEL_PATH = "model.joblib"

def main():
    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError("Run prepare.py before training.")

    df = pd.read_csv(PREPROCESSED_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save model
    joblib.dump(model, MODEL_PATH)

    print("Training complete.")
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
