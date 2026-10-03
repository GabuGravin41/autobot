import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from autobot.core.submission_gate import verify_submission_contract, SubmissionContractViolationError

def test_submission_gate_catches_id_mismatch(tmp_path):
    sample_path = tmp_path / "sample.csv"
    candidate_path = tmp_path / "candidate.csv"

    # Ground truth format: ssp126_0_1
    sample_df = pd.DataFrame({
        "id": ["ssp126_0_1", "ssp126_0_10", "ssp126_0_20"],
        "height": [1.0, 1.0, 1.0],
        "agb": [1.0, 1.0, 1.0]
    })
    sample_df.to_csv(sample_path, index=False)

    # Buggy candidate format: 0_001_ssp126 (the exact bug from Exp 7)
    buggy_df = pd.DataFrame({
        "id": ["0_001_ssp126", "0_010_ssp126", "0_020_ssp126"],
        "height": [1.2, 1.5, 1.8],
        "agb": [0.5, 0.8, 1.1]
    })
    buggy_df.to_csv(candidate_path, index=False)

    # Check ID mismatch
    try:
        verify_submission_contract(candidate_path, sample_path)
        assert False, "Failed to catch ID mismatch!"
    except SubmissionContractViolationError as e:
        assert "CRITICAL ID MISMATCH" in str(e)
        print("TEST PASSED: ID mismatch was intercepted before submission!")

def test_submission_gate_catches_nans(tmp_path):
    sample_path = tmp_path / "sample.csv"
    candidate_path = tmp_path / "candidate.csv"

    sample_df = pd.DataFrame({
        "id": ["ssp126_0_1", "ssp126_0_10"],
        "height": [1.0, 1.0],
        "agb": [1.0, 1.0]
    })
    sample_df.to_csv(sample_path, index=False)

    nan_df = pd.DataFrame({
        "id": ["ssp126_0_1", "ssp126_0_10"],
        "height": [1.2, None],
        "agb": [0.5, 0.8]
    })
    nan_df.to_csv(candidate_path, index=False)

    try:
        verify_submission_contract(candidate_path, sample_path)
        assert False, "Failed to catch NaNs!"
    except SubmissionContractViolationError as e:
        assert "NaN/null values" in str(e)
        print("TEST PASSED: NaN values were intercepted!")

def test_submission_gate_allows_valid(tmp_path):
    sample_path = tmp_path / "sample.csv"
    candidate_path = tmp_path / "candidate.csv"

    sample_df = pd.DataFrame({
        "id": ["ssp126_0_1", "ssp126_0_10"],
        "height": [1.0, 1.0],
        "agb": [1.0, 1.0]
    })
    sample_df.to_csv(sample_path, index=False)

    valid_df = pd.DataFrame({
        "id": ["ssp126_0_1", "ssp126_0_10"],
        "height": [1.2, 1.5],
        "agb": [0.5, 0.8]
    })
    valid_df.to_csv(candidate_path, index=False)

    res = verify_submission_contract(candidate_path, sample_path, min_val=0.0)
    assert res["status"] == "VALID"
    print("TEST PASSED: Valid submission passed gatekeeper.")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_submission_gate_catches_id_mismatch(p)
        test_submission_gate_catches_nans(p)
        test_submission_gate_allows_valid(p)
    print("ALL SUBMISSION GATE TESTS PASSED 100%.")
