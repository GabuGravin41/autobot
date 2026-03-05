import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

class KnowledgeBase:
    """
    Persistent memory for Autobot to store facts, credentials (carefully), 
    and mission-specific data across runs.
    """
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            # Default to root/knowledge_base.json
            root = Path(__file__).resolve().parent.parent
            self.storage_path = root / "knowledge_base.json"
        else:
            self.storage_path = Path(storage_path)
            
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self):
        if self.storage_path.exists():
            try:
                self.data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def save(self):
        try:
            self.storage_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value
        self.save()

    def delete(self, key: str):
        self.data.pop(key, None)
        self.save()

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Simple keyword search across keys and string values."""
        results = []
        q = query.lower()
        for k, v in self.data.items():
            if q in k.lower() or (isinstance(v, str) and q in v.lower()):
                results.append({"key": k, "value": v})
        return results

    def get_all(self) -> Dict[str, Any]:
        return self.data
