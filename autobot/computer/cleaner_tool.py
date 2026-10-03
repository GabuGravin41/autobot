"""
Cleaner Tool — Storage analysis, clutter detection, and disk space management for Autobot.
Integrates with DirCleaner Studio to enforce disk headroom thresholds (>2.0 GB).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Attempt to locate dir_cleaner project
_POSSIBLE_CLEANER_PATHS = [
    Path(__file__).resolve().parents[4] / "dir_cleaner",
    Path(__file__).resolve().parents[3] / "dir_cleaner",
    Path(__file__).resolve().parents[2] / "dir_cleaner",
    Path("c:/Users/User 1/OneDrive/Desktop/projects/django projects/personal projects/dir_cleaner"),
]

for p in _POSSIBLE_CLEANER_PATHS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
        break

try:
    from core.scanner import DirectoryScanner, format_size, KNOWN_BUILD_DIRS
    HAS_DIR_CLEANER = True
except ImportError:
    HAS_DIR_CLEANER = False
    KNOWN_BUILD_DIRS = {"__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", ".mypy_cache"}
    def format_size(bytes_val: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(bytes_val) < 1024.0:
                return f"{bytes_val:3.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} PB"


class Cleaner:
    """Autobot Disk Cleaner and Storage Management Engine."""

    def __init__(self, min_free_gb: float = 2.0):
        self.min_free_gb = min_free_gb

    def get_disk_free(self, path: str = "C:") -> Dict[str, Any]:
        """Check current free and total storage for a drive or directory.
        
        Args:
            path: Drive letter or path to inspect (defaults to 'C:').
            
        Returns:
            Dict containing free_gb, total_gb, percent_free, and headroom_ok.
        """
        try:
            total, used, free = shutil.disk_usage(path)
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            used_gb = used / (1024 ** 3)
            percent_free = (free / total) * 100.0 if total > 0 else 0.0
            return {
                "ok": True,
                "drive": path,
                "free_gb": round(free_gb, 3),
                "used_gb": round(used_gb, 3),
                "total_gb": round(total_gb, 3),
                "percent_free": round(percent_free, 1),
                "headroom_ok": free_gb >= self.min_free_gb,
                "formatted_free": format_size(free),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def scan_path(self, target_path: str = ".", max_items: int = 15) -> Dict[str, Any]:
        """Scan a path for largest files and heaviest subdirectories.
        
        Args:
            target_path: Directory path to scan (defaults to current dir).
            max_items: Maximum items to report in top lists.
            
        Returns:
            Summary of scanned directory bloat and largest elements.
        """
        p = Path(target_path).resolve()
        if not p.exists():
            return {"ok": False, "error": f"Path not found: {p}"}

        if HAS_DIR_CLEANER:
            scanner = DirectoryScanner()
            res = scanner.scan(str(p))
            return {
                "ok": True,
                "path": str(p),
                "total_size": res.total_size,
                "formatted_size": res.formatted_total_size,
                "total_files": res.total_files,
                "total_dirs": res.total_dirs,
                "largest_files": res.largest_files[:max_items],
                "largest_folders": res.largest_folders[:max_items],
                "dense_folders": res.dense_folders[:max_items],
            }

        # Fallback shallow scan
        entries = []
        try:
            for item in p.iterdir():
                try:
                    if item.is_file():
                        entries.append({"name": item.name, "path": str(item), "size": item.stat().st_size, "is_dir": False})
                    elif item.is_dir():
                        sz = sum(f.stat().st_size for f in item.glob("**/*") if f.is_file())
                        entries.append({"name": item.name, "path": str(item), "size": sz, "is_dir": True})
                except Exception:
                    continue
            entries.sort(key=lambda x: x["size"], reverse=True)
            return {
                "ok": True,
                "path": str(p),
                "top_entries": [
                    {**e, "formatted_size": format_size(e["size"])} for e in entries[:max_items]
                ]
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def find_clutter(self, root_dir: str = ".") -> Dict[str, Any]:
        """Find developer caches, build artifacts, and clutter folders.
        
        Args:
            root_dir: Root directory to search from.
            
        Returns:
            List of detected clutter targets with recoverable bytes.
        """
        p = Path(root_dir).resolve()
        clutter_found = []
        total_clutter_bytes = 0

        for r, dirs, files in os.walk(p):
            for d in list(dirs):
                if d in KNOWN_BUILD_DIRS or d.endswith(".egg-info") or d.startswith(".tmp"):
                    full_p = Path(r) / d
                    try:
                        sz = sum(f.stat().st_size for f in full_p.glob("**/*") if f.is_file())
                        clutter_found.append({
                            "type": "cache_directory",
                            "name": d,
                            "path": str(full_p),
                            "bytes": sz,
                            "formatted_size": format_size(sz),
                        })
                        total_clutter_bytes += sz
                        dirs.remove(d) # Don't recurse into found clutter
                    except Exception:
                        pass

        return {
            "ok": True,
            "root": str(p),
            "clutter_count": len(clutter_found),
            "total_recoverable_bytes": total_clutter_bytes,
            "formatted_recoverable": format_size(total_clutter_bytes),
            "items": clutter_found,
        }

    def clean_clutter(self, root_dir: str = ".", dry_run: bool = True) -> Dict[str, Any]:
        """Safely purge detected clutter folders.
        
        Args:
            root_dir: Root directory to search and clean.
            dry_run: If True, only lists what would be cleaned without deleting.
            
        Returns:
            Results of the cleaning operation including freed bytes.
        """
        scan = self.find_clutter(root_dir)
        if not scan.get("ok"):
            return scan

        items = scan["items"]
        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "would_clean_count": len(items),
                "would_recover": scan["formatted_recoverable"],
                "items": items,
            }

        cleaned = []
        failed = []
        freed_bytes = 0

        for item in items:
            path_str = item["path"]
            try:
                p = Path(path_str)
                if p.is_dir():
                    shutil.rmtree(p)
                    cleaned.append(path_str)
                    freed_bytes += item["bytes"]
                elif p.is_file():
                    p.unlink()
                    cleaned.append(path_str)
                    freed_bytes += item["bytes"]
            except Exception as e:
                failed.append({"path": path_str, "error": str(e)})

        return {
            "ok": True,
            "dry_run": False,
            "cleaned_count": len(cleaned),
            "freed_bytes": freed_bytes,
            "formatted_freed": format_size(freed_bytes),
            "failed_count": len(failed),
            "failed": failed,
        }
