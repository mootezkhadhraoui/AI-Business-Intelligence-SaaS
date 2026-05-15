import os
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# =======================
# 🔗 DAGSHUB + MLFLOW
# =======================
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/mootez89/ai-business-intelligence-app.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "mootez89"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c954a7ce876a9580b342a78f6ac3fd93eb32f197"

mlflow.set_experiment("churn_experiment")

# =======================
# 📊 LOAD DATA
# =======================
df = pd.read_csv("data/raw/customer_churn.csv")

# target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# features / target
X = df.drop(columns=["Churn", "customerID"], errors="ignore")
y = df["Churn"]

# =======================
# 🧹 CLEAN + ENCODING
# =======================
X = X.replace({
    "Yes": 1,
    "No": 0,
    "Female": 0,
    "Male": 1
})

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# =======================
# ✂️ SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =======================
# 🤖 MODELS
# =======================
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

best_acc = 0
best_model = None
best_name = ""

# =======================
# 🔁 TRAIN + MLflow
# =======================
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        # logs MLflow
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print(f"✅ {name} accuracy: {acc}")

        # best model
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

print("\n======================")
print(f"🏆 CHAMPION MODEL: {best_name}")
print(f"🎯 ACCURACY: {best_acc}")
print("======================")