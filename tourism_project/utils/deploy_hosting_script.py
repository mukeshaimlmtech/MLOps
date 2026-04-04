from huggingface_hub import HfApi
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("HF_REPO_ID")

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type='space', token=HF_TOKEN, exist_ok=True)
api.upload_folder(folder_path='./',
                  repo_id=REPO_ID,
                  repo_type='space',
                  token=HF_TOKEN)
