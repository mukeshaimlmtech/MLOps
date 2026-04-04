import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("HF_REPO_ID")

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type='dataset', token=HF_TOKEN, exist_ok=True)

dataset_path = hf_hub_download(repo_id=REPO_ID, filename='tourism.csv', repo_type='dataset', token=HF_TOKEN)
df = pd.read_csv(dataset_path)

X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

os.makedirs('tourism_project/model_building/split_data', exist_ok=True)
X_train.to_csv('tourism_project/model_building/split_data/X_train.csv', index=False)
X_test.to_csv('tourism_project/model_building/split_data/X_test.csv', index=False)
y_train.to_csv('tourism_project/model_building/split_data/y_train.csv', index=False)
y_test.to_csv('tourism_project/model_building/split_data/y_test.csv', index=False)

api.upload_folder(folder_path='tourism_project/model_building/split_data',
                  path_in_repo='split_data',
                  repo_id=REPO_ID,
                  repo_type='dataset',
                  token=HF_TOKEN)
