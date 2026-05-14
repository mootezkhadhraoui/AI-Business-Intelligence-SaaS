import dagshub
import mlflow

def init_mlflow():

    dagshub.init(
        repo_owner="mootez89",
        repo_name="ai-business-intelligence-app",
        mlflow=True
    )

    mlflow.set_tracking_uri(
        "https://dagshub.com/mootez89/ai-business-intelligence-app.mlflow"
    )

    mlflow.set_experiment("churn_prediction")