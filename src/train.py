from preprocessing import load_and_preprocess_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = load_and_preprocess_data(
    "../data/raw/customer_churn.csv"
)

models = {

    "RandomForest": RandomForestClassifier(),

    "LogisticRegression": LogisticRegression(
        max_iter=1000
    )
}

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"{name} Accuracy : {acc}")