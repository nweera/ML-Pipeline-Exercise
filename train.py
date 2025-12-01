import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load prepared data
df = pd.read_csv("C:\\Users\\user\\Desktop\\3rd year\\ML-Pipeline-Exercise\\data\\dataset_preprocessed.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "decision_tree": DecisionTreeClassifier(max_depth=5),
    "random_forest": RandomForestClassifier(n_estimators=50),
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Save metrics file per model
    with open(f"metrics_{model_name}.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
