import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import os
from huggingface_hub import HfApi, hf_hub_download

# Set MLflow tracking URI (optional, can be a local directory or remote server)
# For GitHub Actions, it often defaults to a local file-based store, but explicitly setting it can be good practice.
# mlflow.set_tracking_uri("http://localhost:5000") # if you're running a separate MLflow server

# Load data from Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("GITHUB_REPOSITORY_OWNER") + "/Wellness-Tourism-Predictor"

def load_data_from_hf():
    print("Downloading split data from Hugging Face Hub...")
    X_train_path = hf_hub_download(repo_id=REPO_ID, filename='split_data/X_train.csv', repo_type='dataset', token=HF_TOKEN)
    X_test_path = hf_hub_download(repo_id=REPO_ID, filename='split_data/X_test.csv', repo_type='dataset', token=HF_TOKEN)
    y_train_path = hf_hub_download(repo_id=REPO_ID, filename='split_data/y_train.csv', repo_type='dataset', token=HF_TOKEN)
    y_test_path = hf_hub_download(repo_id=REPO_ID, filename='split_data/y_test.csv', repo_type='dataset', token=HF_TOKEN)

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze() # .squeeze() to convert DataFrame to Series if it's a single column
    y_test = pd.read_csv(y_test_path).squeeze()

    print("Data loaded successfully.")
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data_from_hf()

# Identify categorical and numerical features
# Concatenate for feature identification, drop 'Unnamed: 0' if it exists (from CSV export)
X_combined = pd.concat([X_train, X_test])
if 'Unnamed: 0' in X_combined.columns:
    X_combined = X_combined.drop(columns=['Unnamed: 0'])
    X_train = X_train.drop(columns=['Unnamed: 0'], errors='ignore') # errors='ignore' prevents error if column not found
    X_test = X_test.drop(columns=['Unnamed: 0'], errors='ignore')

categorical_features = X_combined.select_dtypes(include=['object', 'category']).columns
numerical_features = X_combined.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models to evaluate
models = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, solver='liblinear'),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0]
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'classifier__max_depth': [None, 10, 20]
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.01, 0.1]
        }
    },
    'XGBClassifier': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.01, 0.1]
        }
    }
}

# Create a directory to store models and preprocessors locally
os.makedirs("tourism_project/model_building/models", exist_ok=True)

best_f1_score = -1
best_model_name = None
best_model_pipeline = None

mlflow.set_experiment("Wellness_Tourism_Package_Prediction")

for model_name, config in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"
Training {model_name}...")

        # Define the full preprocessing and modeling pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])

        # Hyperparameter tuning using GridSearchCV
        print(f"Performing GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(model_pipeline, config['params'], cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Log parameters
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", model_name)

        # Make predictions with the best estimator
        y_pred = best_estimator.predict(X_test)
        y_proba = best_estimator.predict_proba(X_test)[:, 1]

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Best Parameters: {best_params}")

        # Check if this is the best model so far
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = model_name
            best_model_pipeline = best_estimator

        # Log the current model to MLflow
        mlflow.sklearn.log_model(best_estimator, f"model_{model_name.lower()}")

print(f"
Best model found: {best_model_name} with F1-Score: {best_f1_score:.4f}")

# Save the best model locally
if best_model_pipeline:
    model_save_path = f"tourism_project/model_building/models/best_model_{best_model_name.lower()}.joblib"
    joblib.dump(best_model_pipeline, model_save_path)
    print(f"Best model saved locally to: {model_save_path}")

    # Upload the best model to Hugging Face Model Hub
    api = HfApi()
    try:
        # First, upload the model file
        api.upload_file(
            path_or_fileobj=model_save_path,
            path_in_repo=f"best_model_{best_model_name.lower()}.joblib",
            repo_id=REPO_ID,
            repo_type='model',
            token=HF_TOKEN,
        )
        print(f"Best model uploaded to Hugging Face Model Hub: {REPO_ID}/best_model_{best_model_name.lower()}.joblib")

        # Additionally, if you want to upload the preprocessor separately, or as part of the model folder:
        # The preprocessor is part of the best_model_pipeline, so it's already saved with the model.joblib
        # If you needed it standalone for inference services that apply preprocessing steps separately,
        # you'd extract and save/upload it here. For now, it's encapsulated in the pipeline.

    except Exception as e:
        print(f"Error uploading model to Hugging Face: {e}")
else:
    print("No best model found to save/upload.")

