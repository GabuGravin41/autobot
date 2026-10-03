"""
Step 0 — verify Kaggle auth still works.

Run this first. If it fails, the problem is credentials/environment, not
anything below it — stop and fix this before running 01_download.py.
"""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print("AUTHENTICATED OK")

response = api.competitions_list(search="titanic")
comps = getattr(response, "competitions", response)
for c in comps[:5]:
    print(f"  {c.ref}  |  {c.title}")
