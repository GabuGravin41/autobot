"""
Step 1 — download the Titanic competition data.

Titanic - Machine Learning from Disaster is Kaggle's "Getting Started"
competition: no deadline, no prize, effectively unlimited submissions per
day. Picked deliberately as the first live test target because a mistake
here costs nothing.

Downloads train.csv, test.csv, gender_submission.csv into ./data.
"""
import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

os.makedirs("data", exist_ok=True)
api.competition_download_files("titanic", path="data", quiet=False)

# Kaggle sometimes delivers a single titanic.zip instead of loose CSVs.
import zipfile
zip_path = os.path.join("data", "titanic.zip")
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("data")
    print("Extracted titanic.zip")

print("Files in ./data:", os.listdir("data"))
