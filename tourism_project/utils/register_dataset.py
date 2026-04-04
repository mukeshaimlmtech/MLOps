import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("HF_REPO_ID")

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type='dataset', token=HF_TOKEN, exist_ok=True)
api.upload_file(
    path_or_fileobj='tourism_project/data/tourism.csv',
    path_in_repo='tourism.csv',
    repo_id=REPO_ID,
    repo_type='dataset',
    token=HF_TOKEN
)
