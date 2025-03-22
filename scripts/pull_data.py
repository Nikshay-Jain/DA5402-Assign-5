# scripts/pull_data.py
import yaml
import shutil
import os

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

dataset_version = params["data"]["version"]
source_dir = os.path.join("partition", dataset_version)
dest_dir = "data/20k_partition"

# Copy the dataset to the destination folder
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
shutil.copytree(source_dir, dest_dir)

print(f"Pulled dataset version: {dataset_version}")