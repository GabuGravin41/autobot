import requests
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
session = api._load_session()
r = session.get('https://www.kaggle.com/api/v1/competitions/data/list/rsna-knee-abnormality-detection')
if r.status_code == 200:
    data = r.json()
    print('Total files in response:', len(data))
    for item in data:
        name = item.get('name', '')
        if name.endswith('.csv') or name.endswith('.json') or not name.endswith('.dcm'):
            print(f"{name:45s} {item.get('totalBytes', 0):>12,d} bytes")
else:
    print('Failed:', r.status_code, r.text[:200])
