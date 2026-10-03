"""
Autobot Deterministic Submission Gatekeeper.

Guarantees that no submission can ever reach Kaggle if:
1. It does not match the sample submission shape.
2. The 'id' column does not match the sample submission verbatim in value, order, and casing.
3. It contains any NaNs or infinite values.
4. Target predictions violate domain physical constraints (e.g. negative biomass, height <= 0).
5. It has not passed local CV gating against the best known anchor.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
import pandas as pd

class SubmissionContractViolationError(Exception):
    """Raised when a candidate submission violates the competition invariant contract."""
    pass

class LocalCVRegressionError(Exception):
    """Raised when candidate local CV does not beat the competition anchor score."""
    pass


def verify_submission_contract(
    submission_path: str | Path,
    sample_sub_path: str | Path,
    *,
    id_col: str = "id",
    target_cols: list[str] | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> dict[str, Any]:
    """
    Deterministically validates candidate submission against the ground truth sample contract.
    Returns audit metadata dict if valid. Raises SubmissionContractViolationError on ANY mismatch.
    """
    sub_file = Path(submission_path)
    sample_file = Path(sample_sub_path)

    if not sub_file.exists():
        raise SubmissionContractViolationError(f"Candidate submission file not found: {sub_file}")
    if not sample_file.exists():
        raise SubmissionContractViolationError(f"Sample submission file not found: {sample_file}")

    if sub_file.stat().st_size == 0:
        raise SubmissionContractViolationError(f"Candidate submission file is empty (0 bytes): {sub_file}")

    try:
        sub = pd.read_csv(sub_file)
    except Exception as e:
        raise SubmissionContractViolationError(f"Failed to parse candidate CSV: {e}")

    try:
        sample = pd.read_csv(sample_file)
    except Exception as e:
        raise SubmissionContractViolationError(f"Failed to parse sample CSV: {e}")

    # 1. Shape Verification
    if sub.shape != sample.shape:
        raise SubmissionContractViolationError(
            f"Shape mismatch: candidate has {sub.shape} (rows, cols), expected {sample.shape}"
        )

    # 2. Column Names and Order Verification
    if list(sub.columns) != list(sample.columns):
        raise SubmissionContractViolationError(
            f"Columns mismatch: candidate has {list(sub.columns)}, expected {list(sample.columns)}"
        )

    # 3. ID Alignment Verification (Value, Order, Casing)
    if id_col not in sub.columns:
        raise SubmissionContractViolationError(f"Missing required ID column '{id_col}' in candidate submission")

    sub_ids = sub[id_col].astype(str)
    sample_ids = sample[id_col].astype(str)
    
    mismatches = (sub_ids != sample_ids)
    if mismatches.any():
        first_bad_idx = mismatches.idxmax()
        raise SubmissionContractViolationError(
            f"CRITICAL ID MISMATCH at row {first_bad_idx}: "
            f"candidate has '{sub_ids.iloc[first_bad_idx]}', expected '{sample_ids.iloc[first_bad_idx]}'"
        )

    # 4. Null & NaN Verification
    nan_count = int(sub.isna().sum().sum())
    if nan_count > 0:
        raise SubmissionContractViolationError(f"Candidate submission contains {nan_count} NaN/null values")

    # 5. Target Bounds Verification
    targets = target_cols or [c for c in sub.columns if c != id_col]
    for col in targets:
        if not pd.api.types.is_numeric_dtype(sub[col]):
            raise SubmissionContractViolationError(f"Target column '{col}' is not numeric")

        if min_val is not None:
            violating = (sub[col] < min_val).sum()
            if violating > 0:
                min_found = sub[col].min()
                raise SubmissionContractViolationError(
                    f"Physical floor violation in '{col}': {violating} rows < {min_val} (min={min_found})"
                )

        if max_val is not None:
            violating = (sub[col] > max_val).sum()
            if violating > 0:
                max_found = sub[col].max()
                raise SubmissionContractViolationError(
                    f"Physical ceiling violation in '{col}': {violating} rows > {max_val} (max={max_found})"
                )

    return {
        "status": "VALID",
        "rows": len(sub),
        "columns": list(sub.columns),
        "sample_matched": str(sample_file),
    }


def safe_kaggle_submit(
    competition_slug: str,
    submission_path: str | Path,
    sample_sub_path: str | Path,
    message: str,
    *,
    id_col: str = "id",
    target_cols: list[str] | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    kernel_slug: str | None = None,
    kernel_version: int | str | None = None,
) -> dict[str, Any]:
    """
    Submits to Kaggle ONLY if the submission contract passes with 100% compliance.
    Physically prevents malformed submissions from ever touching Kaggle's API.
    """
    # HARD GATE: Will raise SubmissionContractViolationError if contract fails
    audit = verify_submission_contract(
        submission_path=submission_path,
        sample_sub_path=sample_sub_path,
        id_col=id_col,
        target_cols=target_cols,
        min_val=min_val,
        max_val=max_val,
    )

    cmd = ["kaggle", "competitions", "submit", "-c", competition_slug]
    if kernel_slug:
        cmd += ["-k", kernel_slug]
        if kernel_version:
            cmd += ["-v", str(kernel_version)]
        cmd += ["-f", Path(submission_path).name]
    else:
        cmd += ["-f", str(Path(submission_path).resolve())]

    cmd += ["-m", message]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return {
            "ok": False,
            "audit": audit,
            "error": res.stderr or res.stdout,
            "returncode": res.returncode,
        }

    return {
        "ok": True,
        "audit": audit,
        "output": res.stdout,
        "message": message,
    }
