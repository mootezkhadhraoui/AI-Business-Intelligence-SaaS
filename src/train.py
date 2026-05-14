import os
import json
import joblib

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocessing import load_and_preprocess_data
from mlflow_config import init_mlflow


# init MLflow + DagsHub
init_mlflow()

# data
X_train, X_test, y_train, y_test = load_and_preprocess_data(
    "../data/raw/customer_churn.csv"
)

models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=5000, solver="liblinear")
}

best_model = None
best_accuracy = 0
best_name = ""

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print(f"{name} Accuracy: {acc}")

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name


# sauvegarde dossier models
os.makedirs("../models", exist_ok=True)

# champion model
joblib.dump(best_model, "../models/champion_model.pkl")

# features (IMPORTANT pour Streamlit)
with open("../models/features.json", "w") as f:
    json.dump(list(X_train.columns), f)

print("\n🏆 CHAMPION MODEL:", best_name)
print("ACCURACY:", best_accuracy)