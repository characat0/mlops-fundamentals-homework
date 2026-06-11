import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train(data_path, params):
    print(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)

    # ✅ Seleccionar solo columnas numéricas
    X = df.select_dtypes(include=["number"])
    y = df["genre"]

    # ✅ Codificar etiquetas de texto a números
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Crear carpeta models si no existe
    os.makedirs("models", exist_ok=True)

    # Modelo 1: Logistic Regression
    log_reg = LogisticRegression(max_iter=2000)
    log_reg.fit(X, y_encoded)
    log_reg_acc = accuracy_score(y_encoded, log_reg.predict(X))

    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.sklearn.log_model(log_reg, artifact_path="log_reg_model")
        mlflow.log_metric("accuracy", log_reg_acc)

    # Guardar modelo localmente
    joblib.dump(log_reg, "models/log_reg_model.pkl")

    # Modelo 2: XGBoost
    xgb = XGBClassifier(eval_metric="mlogloss")
    xgb.fit(X, y_encoded)
    xgb_acc = accuracy_score(y_encoded, xgb.predict(X))

    with mlflow.start_run(run_name="xgboost"):
        mlflow.sklearn.log_model(xgb, artifact_path="xgb_model")
        mlflow.log_metric("accuracy", xgb_acc)

    # Guardar modelo localmente
    joblib.dump(xgb, "models/xgb_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, required=True)
    args = parser.parse_args()

    params = {}
    train(args.data_path, params)
