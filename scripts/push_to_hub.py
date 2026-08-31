import argparse
from huggingface_hub import HfApi

repo_id = "justinlin722/olmo-1b-gsm8k-grpo"

api = HfApi()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--repo-id",
    default="justinlin722/olmo-1b-gsm8k-grpo",
)

parser.add_argument(
    "--folder",
    default="./outputs/OLMo-2-0425-1B-grpo-v4",
)

args = parser.parse_args()

api.create_repo(
    repo_id=args.repo_id,
    repo_type="model",
    private=True,
    exist_ok=True,
)

api.upload_folder(
    folder_path=args.folder,
    repo_id=repo_id,
    repo_type="model",
)

print("upload model successfully!")
