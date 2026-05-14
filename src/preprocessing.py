import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(path):

    df = pd.read_csv(path)

    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.fillna(
        df.median(numeric_only=True),
        inplace=True
    )

    encoder = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test