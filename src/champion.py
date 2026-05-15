import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import joblib
import os

mlflow.set_tracking_uri(
    "https://dagshub.com/mootez89/ai-business-intelligence-app.mlflow"
)

EXPERIMENT_NAME = "churn_experiment"

MODEL_PATH = "models/champion_model.pkl"


def load_from_mlflow():
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)

    if exp is None:
        return None

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )

    if not runs:
        return None

    run_id = runs[0].info.run_id

    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        return model

    except Exception as e:
        print("MLflow load failed:", e)
        return None


def load_champion_model():
    print("⚙️ Loading champion model...")

    # 1. try MLflow
    model = load_from_mlflow()

    # 2. fallback local (IMPORTANT)
    if model is None:
        if os.path.exists(MODEL_PATH):
            print("⚠️ Using local model fallback")
            model = joblib.load(MODEL_PATH)
        else:
            raise Exception("No model found (MLflow + local failed)")

    return model