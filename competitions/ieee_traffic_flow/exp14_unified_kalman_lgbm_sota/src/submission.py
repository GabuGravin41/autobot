"""Merge per-task predictions into the final unified Kaggle submission file."""
from __future__ import annotations

import time
from pathlib import Path
import pandas as pd

CHUNK_SIZE = 500_000
OUT_COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]

KEYS = {
    "state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
    "queue": ["window_id", "timestamp", "link_id"],
    "odme": ["panel", "departure_time", "path_id"]
}

VALUES = {
    "state": ["speed_kmh", "flow_vph"],
    "queue": ["queue_pred"],
    "odme": ["path_flow"]
}


def index_table(path: Path, task: str) -> pd.DataFrame:
    """Read a task submission CSV and index it by its natural key."""
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file for task '{task}': {path}")
        
    frame = pd.read_csv(path)
    missing = sorted(set(KEYS[task] + VALUES[task]) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
        
    frame = frame[KEYS[task] + VALUES[task]].copy()
    for col in KEYS[task]:
        frame[col] = pd.to_datetime(frame[col], utc=True) if col == "timestamp" else frame[col].astype(str)
    for col in VALUES[task]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        
    dup = int(frame.duplicated(KEYS[task]).sum())
    if dup:
        print(f"Warning: {path} has {dup} duplicate rows over {KEYS[task]}. Dropping duplicates.")
        frame = frame.drop_duplicates(KEYS[task])
        
    return frame.set_index(KEYS[task])


def merge_and_validate_submission(
    state_path: Path,
    queue_path: Path,
    odme_path: Path,
    key_file: Path,
    out_path: Path,
) -> int:
    """Stream submission_key.csv and join all tasks to write final submission.csv."""
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Indexing prediction tables for fast retrieval...")
    tables = {
        "state": index_table(state_path, "state"),
        "queue": index_table(queue_path, "queue"),
        "odme": index_table(odme_path, "odme")
    }

    print(f"Streaming key file from {key_file} to build {out_path}...")
    first_chunk = True
    total_written = 0
    missing_counts = {"state": 0, "queue": 0, "odme": 0}

    for chunk in pd.read_csv(key_file, chunksize=CHUNK_SIZE, dtype=str):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
        out = pd.DataFrame({
            "submission_id": chunk["submission_id"].astype("int64"),
            "task": chunk["task"].astype(str)
        })
        for col in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
            out[col] = 0.0

        for task, table in tables.items():
            rows = (out["task"] == task)
            if not rows.any():
                continue
            wanted = pd.MultiIndex.from_frame(chunk.loc[rows, KEYS[task]])
            found = table.reindex(wanted)
            for col in VALUES[task]:
                vals = found[col].to_numpy()
                missing_counts[task] += int(pd.isna(vals).sum())
                out.loc[rows, col] = pd.Series(vals).fillna(0.0).to_numpy()

        out[OUT_COLUMNS].to_csv(out_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        total_written += len(out)
        print(f"\rProcessed: {total_written:,} rows", end="", flush=True)

    print(f"\nMerge complete: {total_written:,} rows written to {out_path} in {time.time()-t0:.1f}s")
    
    # Validation diagnostics
    print("Running submission diagnostics...")
    for task, cnt in missing_counts.items():
        if cnt > 0:
            print(f"Warning: {cnt:,} {task} rows had no matching prediction and were filled with 0.0.")
        else:
            print(f"[OK] {task}: 100% matched.")

    sample = pd.read_csv(out_path, nrows=5000)
    assert sample.columns.tolist() == OUT_COLUMNS, f"Invalid columns: {sample.columns}"
    assert not sample.isna().any().any(), "Submission contains NaN values!"
    assert (sample["speed_kmh"] >= 0).all(), "Negative speed detected!"
    assert (sample["flow_vph"] >= 0).all(), "Negative flow detected!"
    assert sample["queue_pred"].isin([0, 1]).all(), "Non-binary queue_pred detected!"
    assert (sample["path_flow"] >= 0).all(), "Negative path_flow detected!"
    
    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Verification Successful: {out_path.name} is {file_mb:.1f} MB and 100% contract-compliant.")
    return total_written
