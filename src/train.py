import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import build_preprocessor, load_raw, split

MODEL_NAME = "ChurnModel"
EXPERIMENT_NAME = "telco-churn"

CANDIDATES = [
    {
        "name": "logistic_regression",
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {"C": 1.0, "solver": "lbfgs"},
    },
    {
        "name": "random_forest",
        "model": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "params": {"n_estimators": 200, "max_depth": 10},
    },
    {
        "name": "xgboost",
        "model": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    },
    {
        "name": "xgboost_tuned",
        "model": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=42, eval_metric="logloss", verbosity=0,
        ),
        "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8},
    },
]


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
    }


def main():
    print("Loading data...")
    df = load_raw()
    X_train, X_test, y_train, y_test = split(df)
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Churn rate: {df['Churn'].mean():.1%}\n")

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_run_id = None
    best_roc_auc = 0.0

    for candidate in CANDIDATES:
        with mlflow.start_run(run_name=candidate["name"]) as run:
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", candidate["model"]),
            ])
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)

            mlflow.log_params(candidate["params"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(
                f"{candidate['name']:20s} | "
                f"ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_run_id = run.info.run_id

    # Register best model in MLflow registry with 'champion' alias
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", registered.version)

    # Save locally for Docker — decouples deployment from MLflow tracking server
    output_path = Path(__file__).parent.parent / "model"
    best_pipeline = mlflow.sklearn.load_model(model_uri)
    mlflow.sklearn.save_model(best_pipeline, str(output_path))

    print(f"\nBest run: {best_run_id}")
    print(f"Registered '{MODEL_NAME}' v{registered.version} with alias 'champion' (ROC-AUC: {best_roc_auc:.4f})")
    print(f"Model saved to {output_path}")
    print("\nNext steps:")
    print("  mlflow ui --port 5000   # inspect runs")
    print("  docker build -t churn-api:latest .")
    print("  docker run --rm -p 8000:8000 churn-api:latest")


if __name__ == "__main__":
    main()
