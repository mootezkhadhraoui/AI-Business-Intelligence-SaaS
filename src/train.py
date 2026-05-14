import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/raw/customer_churn.csv")

# CLEAN
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

# encode simple
X = X.replace({"Yes": 1, "No": 0, "Female": 0, "Male": 1})
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

best_acc = 0
best_model = None
best_name = ""

mlflow.set_experiment("churn_experiment")

for name, model in models.items():

    with mlflow.start_run():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        # MLflow logs
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, name)

        print(f"{name} accuracy: {acc}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

# ---------------- SAVE CHAMPION ----------------
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/champion_model.pkl")

print(f"\n🏆 CHAMPION MODEL: {best_name} | ACC: {best_acc}")