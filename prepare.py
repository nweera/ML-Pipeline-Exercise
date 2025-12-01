# prepare.py
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "dataset.csv")
PREPROCESSED_PATH = os.path.join(DATA_DIR, "dataset_preprocessed.csv")

def download_iris(path):
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(path, index=False)
    print(f"Downloaded Iris dataset → {path}")

def preprocess_data(raw_path, out_path):
    print(f"Loading dataset from {raw_path}")
    df = pd.read_csv(raw_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_proc = pd.concat([pd.DataFrame(X_scaled, columns=X.columns), y], axis=1)
    df_proc.to_csv(out_path, index=False)
    print(f"Saved preprocessed dataset → {out_path}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading dataset...")
    download_iris(RAW_PATH)

    print("Preprocessing...")
    preprocess_data(RAW_PATH, PREPROCESSED_PATH)

if __name__ == "__main__":
    main()
