import os
import joblib
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from huggingface_hub import hf_hub_download, HfApi


# =========================================================
# Environment configuration
# =========================================================
HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO = os.environ.get(
    "DATASET_REPO",
    "Mukeshaimlmtech2010/Wellness-Tourism-Dataset",
)
MODEL_REPO = os.environ.get(
    "MODEL_REPO",
    "Mukeshaimlmtech2010/Wellness-Tourism-Model",
)

if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN environment variable is not set")


# =========================================================
# Data loading
# =========================================================
def load_split_data():
    print("Downloading split data from Hugging Face dataset repo")

    X_train = pd.read_csv(
        hf_hub_download(DATASET_REPO, "split_data/X_train.csv",
                        repo_type="dataset", token=HF_TOKEN)
    )
    X_test = pd.read_csv(
        hf_hub_download(DATASET_REPO, "split_data/X_test.csv",
                        repo_type="dataset", token=HF_TOKEN)
    )
    y_train = pd.read_csv(
        hf_hub_download(DATASET_REPO, "split_data/y_train.csv",
                        repo_type="dataset", token=HF_TOKEN)
    ).squeeze()
    y_test = pd.read_csv(
        hf_hub_download(DATASET_REPO, "split_data/y_test.csv",
                        repo_type="dataset", token=HF_TOKEN)
    ).squeeze()

    print("Data loaded successfully")
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_split_data()


# =========================================================
# Feature identification
# =========================================================
combined = pd.concat([X_train, X_test], axis=0)

categorical_features = combined.select_dtypes(include=["object"]).columns.tolist()
numerical_features = combined.select_dtypes(include=["int64", "float64"]).columns.tolist()


# =========================================================
# Preprocessing pipelines
# =========================================================
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# =========================================================
# Model definitions
# =========================================================
model_configs = {
    "logistic_regression": (
        LogisticRegression(solver="liblinear", random_state=42),
        {"classifier__C": [0.1, 1.0, 10.0]},
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=42),
        {"classifier__max_depth": [None, 10, 20]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {"classifier__n_estimators": [100, 200],
         "classifier__max_depth": [None, 10]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {"classifier__n_estimators": [100],
         "classifier__learning_rate": [0.05, 0.1]},
    ),
    "xgboost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {"classifier__n_estimators": [100],
         "classifier__learning_rate": [0.05, 0.1]},
    ),
}


# =========================================================
# Training loop
# =========================================================
mlflow.set_experiment("Wellness_Tourism_Model_Training")

best_f1 = -1.0
best_model_name = None
best_pipeline = None

os.makedirs("tourism_project/model_building/models", exist_ok=True)

for name, (model, params) in model_configs.items():
    print("Starting training for:", name)

    with mlflow.start_run(run_name=name):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        grid = GridSearchCV(
            pipeline,
            params,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        predictions = best_estimator.predict(X_test)
        probabilities = best_estimator.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_estimator, "model")

        print("Metrics:", metrics)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_pipeline = best_estimator


# =========================================================
# Save & upload best model
# =========================================================
print("Best model selected:", best_model_name)
print("Best F1 score:", round(best_f1, 4))

model_path = (
    "tourism_project/model_building/models/"
    + "best_model_"
    + best_model_name
    + ".joblib"
)

joblib.dump(best_pipeline, model_path)

api = HfApi()
api.create_repo(
    repo_id=MODEL_REPO,
    repo_type="model",
    token=HF_TOKEN,
    exist_ok=True,
)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=os.path.basename(model_path),
    repo_id=MODEL_REPO,
    repo_type="model",
    token=HF_TOKEN,
)

print("Best model uploaded successfully to Hugging Face Model Hub")
