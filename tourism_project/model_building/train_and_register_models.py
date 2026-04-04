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
