"""
Kaggle Tool — Wrapper for the official Kaggle API.
"""
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Kaggle:
    """
    Kaggle API wrapper for autonomous competition participation.
    Requires kaggle-api package and ~/.kaggle/kaggle.json credentials.
    """
    
    def __init__(self):
        self._api = None

    def _get_api(self):
        if self._api is None:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                self._api = KaggleApi()
                self._api.authenticate()
            except Exception as e:
                logger.error(f"Kaggle API authentication failed: {e}")
                raise RuntimeError(f"Kaggle API not configured: {e}")
        return self._api

    def list_competitions(self, search: str = None) -> List[Dict[str, Any]]:
        """List active competitions."""
        api = self._get_api()
        comps = api.competitions_list(search=search)
        return [
            {
                "ref": c.ref,
                "title": c.title,
                "description": c.description,
                "deadline": str(c.deadline),
                "category": c.category,
                "reward": c.reward,
            }
            for c in comps
        ]

    def download_data(self, competition: str, path: str = "./data"):
        """Download competition data files."""
        api = self._get_api()
        os.makedirs(path, exist_ok=True)
        api.competition_download_files(competition, path=path, quiet=False)
        logger.info(f"Downloaded data for {competition} to {path}")
        return f"Files downloaded to {path}"

    def submit(self, competition: str, file_path: str, message: str) -> str:
        """Submit a file to a competition."""
        api = self._get_api()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Submission file not found: {file_path}")
        
        api.competition_submit(file_path, message, competition)
        logger.info(f"Submitted {file_path} to {competition}: {message}")
        return f"Successfully submitted to {competition}"

    def get_leaderboard(self, competition: str) -> List[Dict[str, Any]]:
        """Get current leaderboard for a competition."""
        api = self._get_api()
        lb = api.competition_view_leaderboard(competition)
        return [
            {
                "teamName": item.teamName,
                "rank": item.rank,
                "score": item.score,
            }
            for item in lb[:20] # Top 20
        ]
