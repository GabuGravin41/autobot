# --- Cell 1 ---
from __future__ import annotations
import os

os.environ['BIOHUB_MODEL_ARTIFACTS'] = '/kaggle/input/datasets/reyhanksatria/biohub-tracking-support-pack'
os.environ['BIOHUB_TARGET_ARTIFACT_SLUG'] = 'biohub-tracking-support-pack'
os.environ['BIOHUB_ALLOW_ARTIFACT_FALLBACK'] = '1'
os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'] = '/kaggle/input/datasets/reyhanksatria/biohub-deepcenterunet3d-center-prior-v1/weights/full_frame_center/best.pt'
os.environ['BIOHUB_SECONDARY_ARTIFACT_MANIFEST'] = '/kaggle/input/datasets/reyhanksatria/biohub-temporalunet3d-seed-314159-v1/ARTIFACT_MANIFEST.json'

# Decode hexadecimal text into the exact utf-8 source string
def _exact_text(hex_text):
    return bytes.fromhex(hex_text).decode('utf-8')

# Normalize multiline patch text with controlled indentation and trailing newlines
def _patch_text(text, indent = 0, trailing_newline = False):
    prefix = ' ' * indent
    lines = text.splitlines()

    if lines and lines[0].strip().startswith('# '):
        lines = lines[1:]

    if lines and (not lines[0].strip()):
        lines = lines[1:]

    if lines and (not lines[-1].strip()):
        lines = lines[:-1]
    value = chr(10).join((prefix + line if line else '' for line in lines))
    return value + (chr(10) if trailing_newline else '')

import os
from collections import Counter

# Configure the verified production lineage and extend the 0.946 ensemble path
BIOHUB_PRESET = 'harmonic_v3_division_wide'
BIOHUB_SCORE_AXIS = '0.933 baseline -> 0.934 harmonic fusion -> 0.939 wider divisions/calmer fusion -> 0.941 repair adaptation -> 0.946 primary edge-feature TTA -> 0.947 secondary feature TTA + DeepCenter TTA'
os.environ['BIOHUB_OUTPUT_FILTER_SHORT_TRACKS'] = '1'
os.environ['BIOHUB_DET_THRESHOLD'] = '0.965'
os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '6.0'
os.environ['BIOHUB_MOTION_RELINK_RELAXED_UM'] = '10.0'
os.environ['BIOHUB_MOTION_RELINK_LEARNED_BONUS'] = '1.0'
os.environ['BIOHUB_GAP2_MAX_STEP_UM'] = '4.4'
os.environ['BIOHUB_GAP_CLOSE_REUSE_UM'] = '3.2'
os.environ['BIOHUB_ILP_APPEARANCE_WEIGHT'] = '0.0'
os.environ['BIOHUB_ILP_DISAPPEARANCE_WEIGHT'] = '2'
os.environ['BIOHUB_GAP_CLOSE_MAX_GAP'] = '2'
os.environ['BIOHUB_GAP_CLOSE_UM'] = '5.0'
os.environ['BIOHUB_GAP_DENSITY_ADAPTIVE'] = '1'
os.environ['BIOHUB_GAP_DENSITY_REFERENCE_UM'] = '6.5'
os.environ['BIOHUB_GAP_DENSITY_GAIN'] = '0.040'
os.environ['BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM'] = '0.125'
os.environ['BIOHUB_GAP_DENSITY_NEIGHBORS'] = '3'
os.environ['BIOHUB_OUTPUT_MIN_TRACK_LEN'] = '6'
os.environ['BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS'] = '1'
os.environ['BIOHUB_OUTPUT_GAP2_RECOVERY'] = '1'
os.environ['BIOHUB_SAFE_DIV_MAX_UM'] = '9.0'
os.environ['BIOHUB_SAFE_DIV_SISTER_MAX_UM'] = '14.0'
os.environ['BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU'] = '0.6'
os.environ['BIOHUB_SAFE_DIV_DIVERGE_UM'] = '2.25'
os.environ['BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM'] = '10.0'
os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0076'
os.environ['BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP'] = '0.00375'
os.environ['BIOHUB_ILP_DIVISION_WEIGHT'] = '1.2'
os.environ['BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE'] = '1'
os.environ['BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN'] = '4'
os.environ['BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB'] = '0.88'
os.environ['BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM'] = '3.0'
os.environ['BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC'] = '0.012'
os.environ['BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS'] = '120'
os.environ['BIOHUB_USE_DEEPCENTER_VETO'] = '1'
os.environ['BIOHUB_REQUIRE_DEEPCENTER_VETO'] = '1'
os.environ['BIOHUB_DEEPCENTER_EXPECTED_EPOCH'] = '2'
os.environ['BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM'] = '8.5'
os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'] = '/kaggle/input/datasets/reyhanksatria/biohub-deepcenterunet3d-center-prior-v1/weights/full_frame_center/best.pt'
os.environ['BIOHUB_DEEPCENTER_GAP_VETO'] = '1'
os.environ['BIOHUB_DEEPCENTER_GAP_THRESHOLD'] = '0.25'
os.environ['BIOHUB_DEEPCENTER_SAFE_DIV_VETO'] = '1'
os.environ['BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD'] = '0.20'
os.environ['BIOHUB_RUN_OUTPUT_DIAGNOSTICS'] = '0'
os.environ['BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT'] = '0.15'
os.environ['BIOHUB_BIDIRECTIONAL_FUSION_MODE'] = 'harmonic_probability'
os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'
os.environ['BIOHUB_DEEPCENTER_TTA'] = '1'
os.environ['BIOHUB_SECONDARY_EDGE_FEATURE_TTA'] = '1'
os.environ['BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT'] = '0.75'

# Retain the held-out validator and post-process gates from the 0.946 pipeline
os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'
os.environ['BIOHUB_PPSWEEP_SELECT_MARGIN'] = '0.001'
os.environ['BIOHUB_PPSWEEP_MAX_ADJ_LOSS'] = '0.0005'
os.environ['BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION'] = '0.90'
os.environ['BIOHUB_DIAGNOSTIC_ARM'] = 'harmonic_association_production'
print('BIOHUB_PRESET:', BIOHUB_PRESET)
print('BIOHUB_SCORE_AXIS:', BIOHUB_SCORE_AXIS)

import json as _guard_json
import math as _guard_math
import os as _guard_os

# Validate critical environment settings before starting the pipeline
_EXPECTED_NUMERIC = {'BIOHUB_DET_THRESHOLD': 0.965, 'BIOHUB_ILP_APPEARANCE_WEIGHT': 0.0, 'BIOHUB_ILP_DISAPPEARANCE_WEIGHT': 2, 'BIOHUB_GAP_CLOSE_UM': 5.0, 'BIOHUB_OUTPUT_MIN_TRACK_LEN': 6.0, 'BIOHUB_SAFE_DIV_MAX_UM': 9.0, 'BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD': 0.20, 'BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT': 0.15, 'BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT': 0.75}
_EXPECTED_TEXT = {'BIOHUB_BIDIRECTIONAL_FUSION_MODE': 'harmonic_probability', 'BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION': '0.90', 'BIOHUB_EDGE_FEATURE_TTA': '1', 'BIOHUB_DEEPCENTER_TTA': '1', 'BIOHUB_SECONDARY_EDGE_FEATURE_TTA': '1'}
_drift = {}

for _key, _want in _EXPECTED_NUMERIC.items():
    _raw = _guard_os.environ.get(_key)

    if _raw is None:
        _drift[_key] = 'missing'
        continue
    _got = float(_raw)

    if not _guard_math.isclose(_got, _want, rel_tol = 0.0, abs_tol = 1e-12):
        _drift[_key] = {'expected': _want, 'actual': _got}

for _key, _want in _EXPECTED_TEXT.items():
    _got = _guard_os.environ.get(_key)

    if _got != _want:
        _drift[_key] = {'expected': _want, 'actual': _got}

if _drift:
    raise RuntimeError('Configuration drift detected: ' + _guard_json.dumps(_drift, sort_keys = True))

print('Configuration guard: PASS')
print('Verified score progression: 0.933 -> 0.934 -> 0.939 -> 0.941 -> 0.946 -> 0.947')
print('Previous verified configuration: public LB 0.946')
print('Current verified configuration: public LB 0.947')
print('0.946 primary edge-feature TTA retained with eight aligned D4 views')
print('0.947 change: secondary feature TTA blend 0.75 + DeepCenter TTA + safe-div threshold 0.20')
print('Reverse-time association weight retained at 0.150')

import csv
import contextlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
import sys
import threading
import time
from pathlib import Path
import pandas as pd
from IPython.display import display

# Resolve kaggle competition paths and initialize output locations
COMPETITION = 'biohub-cell-tracking-during-development'
COMP_DIR_CANDIDATES = [Path(f'/kaggle/input/competitions/{COMPETITION}'), Path(f'/kaggle/input/{COMPETITION}')]
COMP_DIR = next((path for path in COMP_DIR_CANDIDATES if path.exists()), COMP_DIR_CANDIDATES[0])
TEST_DIR = COMP_DIR / 'test'
WORKING_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
REPO_DIR = WORKING_DIR / 'tracking_repo'
SUBMISSION_PATH = WORKING_DIR / 'submission.csv'
RUN_STATS_PATH = WORKING_DIR / 'run_stats.csv'
METHOD = 'unet_transformer'
WEIGHTS_RELATIVE = f'weights/{METHOD}/split_0/edge_predictor_best.pth'
EXPERIMENT_TAG = 'secondary_deepcenter_tta_0947'
RESUME_DIR = WORKING_DIR / 'biohub_live_resume'
TEST_PREDICTION_STATE_PATH = RESUME_DIR / 'test_prediction_state.json'
TEST_PREDICTION_LOG_PATH = RESUME_DIR / 'test_prediction.log'
BASE_SUBMISSION_STATE_PATH = RESUME_DIR / 'base_submission_state.json'
BASE_SUBMISSION_LOG_PATH = RESUME_DIR / 'base_submission.log'
VALIDATOR_PREDICTION_STATE_PATH = RESUME_DIR / 'validator_prediction_state.json'
VALIDATOR_PREDICTION_LOG_PATH = RESUME_DIR / 'validator_prediction.log'
VALIDATOR_BASE_STATE_PATH = RESUME_DIR / 'validator_base_state.json'
VALIDATOR_BASE_LOG_PATH = RESUME_DIR / 'validator_base.log'
PPSWEEP_STATE_PATH = RESUME_DIR / 'ppsweep_state.json'
PPSWEEP_LOG_PATH = RESUME_DIR / 'ppsweep.log'
FINAL_SUBMISSION_STATE_PATH = RESUME_DIR / 'final_submission_state.json'
FINAL_SUBMISSION_LOG_PATH = RESUME_DIR / 'final_submission.log'
RESUME_PREDICTIONS_PATH = RESUME_DIR / 'predictions'
RESUME_SCHEMA_VERSION = 'biohub_0947_live_resume_v1'
RESUME_DIR.mkdir(parents = True, exist_ok = True)

# Convert cached state values into json-safe python objects
def _resume_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _resume_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_resume_json_safe(item) for item in value]

    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return value

# Write resume metadata atomically so incomplete stages never look complete
def _write_resume_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    temp_path = path.with_name(path.name + '.tmp')
    temp_path.write_text(json.dumps(_resume_json_safe(payload), indent = 2, sort_keys = True) + '\n', encoding = 'utf-8')
    temp_path.replace(path)

# Read a completed resume metadata file when it is valid json
def _read_resume_json(path: Path) -> dict | None:
    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding = 'utf-8'))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None

# Remove a cached file when a dependent stage must be recomputed
def _unlink_resume_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

# Compute a stable file checksum for resume validation
def _resume_file_sha256(path: Path) -> str:
    import hashlib as _resume_hashlib
    digest = _resume_hashlib.sha256()

    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

# Count csv data rows without loading the full file into memory
def _resume_csv_row_count(path: Path) -> int:
    with path.open('r', encoding = 'utf-8') as handle:
        return max(sum((1 for _ in handle)) - 1, 0)

# Compute a stable signature for cached stage inputs
def _resume_signature(payload: dict) -> str:
    import hashlib as _resume_hashlib
    encoded = json.dumps(_resume_json_safe(payload), sort_keys = True, separators = (',', ':')).encode('utf-8')
    return _resume_hashlib.sha256(encoded).hexdigest()

# Return prediction graphs for one exact method name
def _resume_prediction_paths(method: str) -> list[Path]:
    return sorted((REPO_DIR / 'predictions').glob(f'*/{method}/split_0/*.geff'))

# Verify that a cached prediction method covers exactly the expected datasets
def _resume_prediction_complete(method: str, expected_stems: list[str]) -> bool:
    paths = _resume_prediction_paths(method)
    stems = [path.stem for path in paths]
    return len(paths) == len(expected_stems) and len(stems) == len(set(stems)) and set(stems) == set(expected_stems)

# Verify that cached frame-retention diagnostics cover every expected test movie
def _resume_retention_guard_complete(expected_stems: list[str]) -> bool:
    expected = set(expected_stems)
    keys = []
    movies = set()

    try:
        for path in sorted(WORKING_DIR.glob('retention_guard_*.jsonl')):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                movie = str(record.get('dataset', ''))

                if movie not in expected:
                    continue
                key = (movie, int(record['frame']))
                keys.append(key)
                movies.add(movie)
    except Exception:
        return False
    return bool(keys) and movies == expected and len(keys) == len(set(keys))

# Remove stale prediction methods without touching unrelated completed methods
def _remove_prediction_methods(method_prefix: str) -> None:
    predictions_root = REPO_DIR / 'predictions'

    if not predictions_root.exists():
        return

    for username_root in predictions_root.iterdir():
        if not username_root.is_dir():
            continue

        for method_root in list(username_root.iterdir()):
            if method_root.name == method_prefix or method_root.name.startswith(method_prefix + '_gpu'):
                remove_path(method_root)

# Build a post-processing signature from the active production constants
def _postprocess_resume_signature() -> str:
    prefixes = ('OUTPUT_', 'MOTION_', 'GAP_', 'GAP2_', 'SAFE_DIV_', 'DEEPCENTER_', 'SHORT_TRACK_', 'DIV_')
    config = {}

    for key, value in globals().items():
        if key.startswith(prefixes) and isinstance(value, (str, int, float, bool, type(None))):
            config[key] = value
    return _resume_signature({'resume_schema': RESUME_SCHEMA_VERSION, 'experiment_tag': EXPERIMENT_TAG, 'config': config})

# Duplicate live notebook output into a stage log for exact replay on resume
class _TeeTextIO:
    def __init__(self, streams, lock):
        self.streams = streams
        self.lock = lock

    def write(self, text):
        with self.lock:
            for stream in self.streams:
                stream.write(text)
        return len(text)

    def flush(self):
        with self.lock:
            for stream in self.streams:
                stream.flush()

    def isatty(self):
        return bool(getattr(self.streams[0], 'isatty', lambda: False)())

    def fileno(self):
        return self.streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self.streams[0], 'encoding', 'utf-8')

    def __getattr__(self, name):
        return getattr(self.streams[0], name)

# Capture one expensive stage while keeping its normal notebook output visible
def _capture_stage_output(log_path: Path):
    @contextlib.contextmanager
    def _manager():
        log_path.parent.mkdir(parents = True, exist_ok = True)
        lock = threading.Lock()

        with log_path.open('w', encoding = 'utf-8') as log_handle:
            stdout_tee = _TeeTextIO((sys.stdout, log_handle), lock)
            stderr_tee = _TeeTextIO((sys.stderr, log_handle), lock)

            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                yield
    return _manager()

# Replay a completed stage with the same cached console output
def _replay_stage_output(log_path: Path) -> None:
    if log_path.is_file():
        sys.stdout.write(log_path.read_text(encoding = 'utf-8'))
        sys.stdout.flush()

# Stream subprocess output into the notebook so stage capture remains complete
def _popen_streamed(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    stream_env = dict(env)
    stream_env.setdefault('PYTHONUNBUFFERED', '1')
    process = subprocess.Popen(command, cwd = cwd, env = stream_env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, bufsize = 1)

    def _pump_output() -> None:
        if process.stdout is None:
            return

        for line in process.stdout:
            print(line, end = '', flush = True)

    thread = threading.Thread(target = _pump_output, daemon = True)
    thread.start()
    process._biohub_output_thread = thread
    return process

# Join the output thread attached to a streamed subprocess
def _join_streamed_process(process: subprocess.Popen) -> None:
    thread = getattr(process, '_biohub_output_thread', None)

    if thread is not None:
        thread.join()

# Run one streamed subprocess and preserve check-true behavior
def _run_subprocess_streamed(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    process = _popen_streamed(command, cwd = cwd, env = env)
    return_code = process.wait()
    _join_streamed_process(process)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
        
TARGET_ARTIFACT_SLUG = os.environ.get('BIOHUB_TARGET_ARTIFACT_SLUG', 'biohub-tracking-support-pack-50ep-v1')
PRIMARY_ARTIFACT_MANIFEST = Path(os.environ.get('BIOHUB_PRIMARY_ARTIFACT_MANIFEST', '/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json'))
ALLOW_ARTIFACT_FALLBACK = os.environ.get('BIOHUB_ALLOW_ARTIFACT_FALLBACK', '0') != '0'
DET_THRESHOLD = float(os.environ.get('BIOHUB_DET_THRESHOLD', '0.99'))
UNET_BATCH_SIZE = int(os.environ.get('BIOHUB_UNET_BATCH_SIZE', '4'))
USE_ILP = os.environ.get('BIOHUB_USE_ILP', '1') != '0'
ILP_EDGE_WEIGHT = float(os.environ.get('BIOHUB_ILP_EDGE_WEIGHT', '-1.0'))
ILP_APPEARANCE_WEIGHT = float(os.environ.get('BIOHUB_ILP_APPEARANCE_WEIGHT', '0.1'))
ILP_DISAPPEARANCE_WEIGHT = float(os.environ.get('BIOHUB_ILP_DISAPPEARANCE_WEIGHT', '0.1'))
ILP_DIVISION_WEIGHT = float(os.environ.get('BIOHUB_ILP_DIVISION_WEIGHT', '1.0'))
SLICE = ''
ALLOW_PIP_INSTALL = os.environ.get('BIOHUB_ALLOW_PIP_INSTALL', '0') != '0'
RUN_OUTPUT_DIAGNOSTICS = os.environ.get('BIOHUB_RUN_OUTPUT_DIAGNOSTICS', '1') != '0'
OUTPUT_EDGE_MAX_UM = float(os.environ.get('BIOHUB_OUTPUT_EDGE_MAX_UM', '14.0'))
OUTPUT_ENFORCE_NEXT_FRAME = os.environ.get('BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME', '1') != '0'
OUTPUT_SINGLE_PARENT_REPAIR = os.environ.get('BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR', '1') != '0'
OUTPUT_SINGLE_CHILD_REPAIR = os.environ.get('BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR', '0') != '0'
OUTPUT_PRUNE_ISOLATED = os.environ.get('BIOHUB_OUTPUT_PRUNE_ISOLATED', '1') != '0'
OUTPUT_MOTION_RELINK = os.environ.get('BIOHUB_OUTPUT_MOTION_RELINK', '1') != '0'
MOTION_RELINK_TIGHT_UM = float(os.environ.get('BIOHUB_MOTION_RELINK_TIGHT_UM', '6.0'))
MOTION_RELINK_RELAXED_UM = float(os.environ.get('BIOHUB_MOTION_RELINK_RELAXED_UM', '10.0'))
MOTION_RELINK_VELOCITY_WEIGHT = float(os.environ.get('BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT', '0.5'))
MOTION_RELINK_LEARNED_BONUS = float(os.environ.get('BIOHUB_MOTION_RELINK_LEARNED_BONUS', '0.75'))
MOTION_RELINK_MAX_FRAME_NODES = int(os.environ.get('BIOHUB_MOTION_RELINK_MAX_FRAME_NODES', '2600'))
OUTPUT_DIVISION_GEOMETRY_FILTER = os.environ.get('BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER', '0') != '0'
DIV_PARENT_MAX_UM = float(os.environ.get('BIOHUB_DIV_PARENT_MAX_UM', '10.5'))
DIV_SISTER_MAX_UM = float(os.environ.get('BIOHUB_DIV_SISTER_MAX_UM', '8.0'))
DIV_DROP_TO_SINGLE_IF_BAD = os.environ.get('BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD', '1') != '0'
OUTPUT_GAP_CLOSE = os.environ.get('BIOHUB_OUTPUT_GAP_CLOSE', '1') != '0'
GAP_CLOSE_MAX_GAP = int(os.environ.get('BIOHUB_GAP_CLOSE_MAX_GAP', '1'))
GAP_CLOSE_UM = float(os.environ.get('BIOHUB_GAP_CLOSE_UM', '6.0'))
GAP_DENSITY_ADAPTIVE = os.environ.get('BIOHUB_GAP_DENSITY_ADAPTIVE', '0') != '0'
GAP_DENSITY_REFERENCE_UM = float(os.environ.get('BIOHUB_GAP_DENSITY_REFERENCE_UM', '6.5'))
GAP_DENSITY_GAIN = float(os.environ.get('BIOHUB_GAP_DENSITY_GAIN', '0.040'))
GAP_DENSITY_MAX_STEP_DELTA_UM = float(os.environ.get('BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM', '0.125'))
GAP_DENSITY_NEIGHBORS = int(os.environ.get('BIOHUB_GAP_DENSITY_NEIGHBORS', '3'))
GAP_CLOSE_REUSE_EXISTING = os.environ.get('BIOHUB_GAP_CLOSE_REUSE_EXISTING', '1') != '0'
GAP_CLOSE_REUSE_UM = float(os.environ.get('BIOHUB_GAP_CLOSE_REUSE_UM', '3.2'))
GAP_CLOSE_MAX_ADDED_FRAC = float(os.environ.get('BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC', '0.05'))
GAP_CLOSE_MAX_ADDED_ABS = int(os.environ.get('BIOHUB_GAP_CLOSE_MAX_ADDED_ABS', '2000'))
GAP_REFINE_SYNTHETIC = os.environ.get('BIOHUB_GAP_REFINE_SYNTHETIC', '1') != '0'
GAP_REFINE_WIN_Z = int(os.environ.get('BIOHUB_GAP_REFINE_WIN_Z', '1'))
GAP_REFINE_WIN_YX = int(os.environ.get('BIOHUB_GAP_REFINE_WIN_YX', '3'))
GAP_REFINE_MAX_SHIFT_UM = float(os.environ.get('BIOHUB_GAP_REFINE_MAX_SHIFT_UM', '3.2'))
OUTPUT_FILTER_SHORT_TRACKS = os.environ.get('BIOHUB_OUTPUT_FILTER_SHORT_TRACKS', '1') != '0'
OUTPUT_MIN_TRACK_LEN = int(os.environ.get('BIOHUB_OUTPUT_MIN_TRACK_LEN', '6'))
OUTPUT_KEEP_DIVISION_COMPONENTS = os.environ.get('BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS', '1') != '0'
ADAPTIVE_SHORT_TRACK_RESCUE = os.environ.get('BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE', '0') != '0'
SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC = float(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC', '0.10'))
SHORT_TRACK_RESCUE_MIN_LEN = int(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN', '4'))
SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB = float(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB', '0.82'))
SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM = float(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM', '3.25'))
SHORT_TRACK_RESCUE_MAX_NODES_FRAC = float(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC', '0.018'))
SHORT_TRACK_RESCUE_MAX_NODES_ABS = int(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS', '180'))
OUTPUT_LINEFIT_SMOOTH = os.environ.get('BIOHUB_OUTPUT_LINEFIT_SMOOTH', '1') != '0'
OUTPUT_LINEFIT_WEIGHT = float(os.environ.get('BIOHUB_OUTPUT_LINEFIT_WEIGHT', '0.8'))
OUTPUT_LINEFIT_WINDOW = int(os.environ.get('BIOHUB_OUTPUT_LINEFIT_WINDOW', '2'))
OUTPUT_GAP2_RECOVERY = os.environ.get('BIOHUB_OUTPUT_GAP2_RECOVERY', '0') != '0'
GAP2_MAX_TOTAL_UM = float(os.environ.get('BIOHUB_GAP2_MAX_TOTAL_UM', '10.2'))
GAP2_MAX_STEP_UM = float(os.environ.get('BIOHUB_GAP2_MAX_STEP_UM', '4.4'))
GAP2_MAX_LINKS_FRAC = float(os.environ.get('BIOHUB_GAP2_MAX_LINKS_FRAC', '0.0045'))
GAP2_MAX_LINKS_ABS = int(os.environ.get('BIOHUB_GAP2_MAX_LINKS_ABS', '180'))
GAP2_REQUIRE_CONTEXT = os.environ.get('BIOHUB_GAP2_REQUIRE_CONTEXT', '1') != '0'
GAP2_FRAME_FRAC_CAP = float(os.environ.get('BIOHUB_GAP2_FRAME_FRAC_CAP', '0.006'))
OUTPUT_SAFE_DIVISIONS = os.environ.get('BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'
SAFE_DIV_MAX_UM = float(os.environ.get('BIOHUB_SAFE_DIV_MAX_UM', '4.7'))
SAFE_DIV_SISTER_MAX_UM = float(os.environ.get('BIOHUB_SAFE_DIV_SISTER_MAX_UM', '7.2'))
SAFE_DIV_SISTER_SYMMETRY_TAU = float(os.environ.get('BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU', '0.0'))
SAFE_DIV_EXISTING_CHILD_MAX_UM = float(os.environ.get('BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM', '7.8'))
SAFE_DIV_FRAME_FRAC_CAP = float(os.environ.get('BIOHUB_SAFE_DIV_FRAME_FRAC_CAP', '0.008'))
SAFE_DIV_GLOBAL_FRAC_CAP = float(os.environ.get('BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP', '0.004'))
SAFE_DIV_DIVERGE_UM = float(os.environ.get('BIOHUB_SAFE_DIV_DIVERGE_UM', '2.25'))
SAFE_DIV_REQUIRE_DIVERGENCE = os.environ.get('BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE', '1') != '0'
SAFE_DIV_REQUIRE_MUTUAL_NN = os.environ.get('BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN', '1') != '0'
USE_DEEPCENTER_VETO = os.environ.get('BIOHUB_USE_DEEPCENTER_VETO', '1') != '0'
REQUIRE_DEEPCENTER_VETO = os.environ.get('BIOHUB_REQUIRE_DEEPCENTER_VETO', '1') != '0'
DEEPCENTER_MANIFEST_DEFAULT = os.environ.get('BIOHUB_DEEPCENTER_MANIFEST_DEFAULT', '/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/ARTIFACT_MANIFEST.json')
DEEPCENTER_CHECKPOINT_DEFAULT = os.environ.get('BIOHUB_DEEPCENTER_CHECKPOINT_DEFAULT', '/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt')
DEEPCENTER_RELATIVE = os.environ.get('BIOHUB_DEEPCENTER_RELATIVE', 'weights/full_frame_center/best.pt')
DEEPCENTER_GAP_VETO = os.environ.get('BIOHUB_DEEPCENTER_GAP_VETO', '1') != '0'
DEEPCENTER_SAFE_DIV_VETO = os.environ.get('BIOHUB_DEEPCENTER_SAFE_DIV_VETO', '1') != '0'
DEEPCENTER_GAP_THRESHOLD = float(os.environ.get('BIOHUB_DEEPCENTER_GAP_THRESHOLD', '0.10'))
DEEPCENTER_EXPECTED_EPOCH = int(os.environ.get('BIOHUB_DEEPCENTER_EXPECTED_EPOCH', '0'))
DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM = float(os.environ.get('BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM', '0'))
DEEPCENTER_SAFE_DIV_THRESHOLD = float(os.environ.get('BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD', '0.12'))
DEEPCENTER_SCORE_WIN_Z = int(os.environ.get('BIOHUB_DEEPCENTER_SCORE_WIN_Z', '1'))
DEEPCENTER_SCORE_WIN_YX = int(os.environ.get('BIOHUB_DEEPCENTER_SCORE_WIN_YX', '2'))
DEEPCENTER_SCORE_CACHE_MAX_FRAMES = int(os.environ.get('BIOHUB_DEEPCENTER_SCORE_CACHE_MAX_FRAMES', '8'))
CONFIG_DISPLAY = {'experiment_tag': EXPERIMENT_TAG, 'method': METHOD, 'weights': WEIGHTS_RELATIVE, 'target_artifact_slug': TARGET_ARTIFACT_SLUG, 'primary_artifact_manifest': str(PRIMARY_ARTIFACT_MANIFEST), 'allow_artifact_fallback': ALLOW_ARTIFACT_FALLBACK, 'det_threshold': DET_THRESHOLD, 'unet_batch_size': UNET_BATCH_SIZE, 'use_ilp': USE_ILP, 'ilp_edge_weight': ILP_EDGE_WEIGHT, 'ilp_appearance_weight': ILP_APPEARANCE_WEIGHT, 'ilp_disappearance_weight': ILP_DISAPPEARANCE_WEIGHT, 'ilp_division_weight': ILP_DIVISION_WEIGHT, 'slice': SLICE, 'allow_pip_install': ALLOW_PIP_INSTALL, 'output_edge_max_um': OUTPUT_EDGE_MAX_UM, 'output_enforce_next_frame': OUTPUT_ENFORCE_NEXT_FRAME, 'output_single_parent_repair': OUTPUT_SINGLE_PARENT_REPAIR, 'output_single_child_repair': OUTPUT_SINGLE_CHILD_REPAIR, 'output_prune_isolated': OUTPUT_PRUNE_ISOLATED, 'output_motion_relink': OUTPUT_MOTION_RELINK, 'motion_relink_tight_um': MOTION_RELINK_TIGHT_UM, 'motion_relink_relaxed_um': MOTION_RELINK_RELAXED_UM, 'motion_relink_velocity_weight': MOTION_RELINK_VELOCITY_WEIGHT, 'motion_relink_learned_bonus': MOTION_RELINK_LEARNED_BONUS, 'motion_relink_max_frame_nodes': MOTION_RELINK_MAX_FRAME_NODES, 'output_division_geometry_filter': OUTPUT_DIVISION_GEOMETRY_FILTER, 'div_parent_max_um': DIV_PARENT_MAX_UM, 'div_sister_max_um': DIV_SISTER_MAX_UM, 'div_drop_to_single_if_bad': DIV_DROP_TO_SINGLE_IF_BAD, 'output_gap_close': OUTPUT_GAP_CLOSE, 'gap_close_max_gap': GAP_CLOSE_MAX_GAP, 'gap_close_effective_max_gap': min(GAP_CLOSE_MAX_GAP, 1), 'gap_close_um': GAP_CLOSE_UM, 'gap_density_adaptive': GAP_DENSITY_ADAPTIVE, 'gap_density_reference_um': GAP_DENSITY_REFERENCE_UM, 'gap_density_gain': GAP_DENSITY_GAIN, 'gap_density_max_step_delta_um': GAP_DENSITY_MAX_STEP_DELTA_UM, 'gap_density_neighbors': GAP_DENSITY_NEIGHBORS, 'gap_close_reuse_existing': GAP_CLOSE_REUSE_EXISTING, 'gap_close_reuse_um': GAP_CLOSE_REUSE_UM, 'gap_close_max_added_frac': GAP_CLOSE_MAX_ADDED_FRAC, 'gap_close_max_added_abs': GAP_CLOSE_MAX_ADDED_ABS, 'gap_refine_synthetic': GAP_REFINE_SYNTHETIC, 'gap_refine_win_z': GAP_REFINE_WIN_Z, 'gap_refine_win_yx': GAP_REFINE_WIN_YX, 'gap_refine_max_shift_um': GAP_REFINE_MAX_SHIFT_UM, 'output_filter_short_tracks': OUTPUT_FILTER_SHORT_TRACKS, 'output_min_track_len': OUTPUT_MIN_TRACK_LEN, 'output_keep_division_components': OUTPUT_KEEP_DIVISION_COMPONENTS, 'adaptive_short_track_rescue': ADAPTIVE_SHORT_TRACK_RESCUE, 'short_track_rescue_trigger_removed_frac': SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC, 'short_track_rescue_min_len': SHORT_TRACK_RESCUE_MIN_LEN, 'short_track_rescue_min_mean_edge_prob': SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB, 'short_track_rescue_max_mean_edge_dist_um': SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM, 'short_track_rescue_max_nodes_frac': SHORT_TRACK_RESCUE_MAX_NODES_FRAC, 'short_track_rescue_max_nodes_abs': SHORT_TRACK_RESCUE_MAX_NODES_ABS, 'output_linefit_smooth': OUTPUT_LINEFIT_SMOOTH, 'output_linefit_weight': OUTPUT_LINEFIT_WEIGHT, 'output_linefit_window': OUTPUT_LINEFIT_WINDOW, 'output_gap2_recovery': OUTPUT_GAP2_RECOVERY, 'gap2_max_total_um': GAP2_MAX_TOTAL_UM, 'gap2_max_step_um': GAP2_MAX_STEP_UM, 'gap2_max_links_frac': GAP2_MAX_LINKS_FRAC, 'gap2_max_links_abs': GAP2_MAX_LINKS_ABS, 'gap2_require_context': GAP2_REQUIRE_CONTEXT, 'gap2_frame_frac_cap': GAP2_FRAME_FRAC_CAP, 'output_safe_divisions': OUTPUT_SAFE_DIVISIONS, 'safe_div_max_um': SAFE_DIV_MAX_UM, 'safe_div_sister_max_um': SAFE_DIV_SISTER_MAX_UM, 'safe_div_existing_child_max_um': SAFE_DIV_EXISTING_CHILD_MAX_UM, 'safe_div_frame_frac_cap': SAFE_DIV_FRAME_FRAC_CAP, 'safe_div_global_frac_cap': SAFE_DIV_GLOBAL_FRAC_CAP, 'use_deepcenter_add_only_gate': USE_DEEPCENTER_VETO, 'deepcenter_gap_add_gate': DEEPCENTER_GAP_VETO, 'deepcenter_safe_div_add_gate': DEEPCENTER_SAFE_DIV_VETO, 'deepcenter_gap_threshold': DEEPCENTER_GAP_THRESHOLD, 'deepcenter_expected_epoch': DEEPCENTER_EXPECTED_EPOCH, 'deepcenter_gap_confirm_min_span_um': DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM, 'deepcenter_safe_div_threshold': DEEPCENTER_SAFE_DIV_THRESHOLD, 'deepcenter_tta': os.environ.get('BIOHUB_DEEPCENTER_TTA', '0') != '0', 'secondary_edge_feature_tta': os.environ.get('BIOHUB_SECONDARY_EDGE_FEATURE_TTA', '0') != '0', 'secondary_edge_feature_tta_weight': float(os.environ.get('BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT', '0.75')), 'deepcenter_checkpoint_default': DEEPCENTER_CHECKPOINT_DEFAULT}
print('Biohub learned UNet + node-transformer + ILP submission')
print('COMP_DIR:', COMP_DIR, 'exists:', COMP_DIR.exists())
print('TEST_DIR:', TEST_DIR, 'exists:', TEST_DIR.exists())
print(json.dumps(CONFIG_DISPLAY, indent = 2, sort_keys = True))

import re

os.environ.setdefault('POLARS_PREFER_PKG', '32')

# Prepare required runtime dependencies and offline installation metadata
PACKAGE_SPECS = {'tracksdata': ('tracksdata', 'tracksdata'), 'zarr': ('zarr', 'zarr>=3.0.10,<4'), 'pyscipopt': ('pyscipopt', 'pyscipopt'), 'geff': ('geff', 'geff>=1.1.3.1.1'), 'geff_spec': ('geff_spec', 'geff-spec<1.2'), 'ilpy': ('ilpy', 'ilpy>=0.5.1'), 'polars': ('polars', 'polars>=1.36'), 'blosc2': ('blosc2', 'blosc2'), 'dask': ('dask', 'dask'), 'imagecodecs': ('imagecodecs', 'imagecodecs'), 'skimage': ('skimage', 'scikit-image>=0.24'), 'pyarrow': ('pyarrow', 'pyarrow'), 'rustworkx': ('rustworkx', 'rustworkx>=0.17.1'), 'sqlalchemy': ('sqlalchemy', 'sqlalchemy>=2'), 'numcodecs': ('numcodecs', 'numcodecs>=0.13,<0.16'), 'donfig': ('donfig', 'donfig>=0.8'), 'google_crc32c': ('google_crc32c', 'google-crc32c>=1.5'), 'bidict': ('bidict', 'bidict>=0.23.1'), 'psygnal': ('psygnal', 'psygnal>=0.14'), 'rich': ('rich', 'rich'), 'networkx': ('networkx', 'networkx>=3.2.1'), 'pydantic': ('pydantic', 'pydantic>=2.11'), 'pydantic_core': ('pydantic_core', 'pydantic-core'), 'annotated_types': ('annotated_types', 'annotated-types'), 'typing_extensions': ('typing_extensions', 'typing-extensions>=4.13'), 'typing_inspection': ('typing_inspection', 'typing-inspection'), 'markdown_it': ('markdown_it', 'markdown-it-py'), 'pygments': ('pygments', 'pygments'), 'click': ('click', 'click'), 'cloudpickle': ('cloudpickle', 'cloudpickle'), 'fsspec': ('fsspec', 'fsspec'), 'partd': ('partd', 'partd'), 'locket': ('locket', 'locket'), 'toolz': ('toolz', 'toolz'), 'yaml': ('yaml', 'pyyaml'), 'ndindex': ('ndindex', 'ndindex'), 'msgpack': ('msgpack', 'msgpack'), 'numexpr': ('numexpr', 'numexpr'), 'deprecated': ('deprecated', 'deprecated'), 'wrapt': ('wrapt', 'wrapt'), 'imageio': ('imageio', 'imageio'), 'PIL': ('PIL', 'pillow'), 'tifffile': ('tifffile', 'tifffile'), 'lazy_loader': ('lazy_loader', 'lazy-loader'), 'tqdm': ('tqdm', 'tqdm')}
EXTRA_SPECS_BY_NAME = {'tracksdata': ['bidict>=0.23.1', 'psygnal>=0.14', 'rich'], 'zarr': ['donfig>=0.8', 'google-crc32c>=1.5', 'numcodecs>=0.13,<0.16'], 'geff': ['geff-spec<1.2', 'networkx>=3.2.1', 'pydantic>=2.11', 'numcodecs>=0.13,<0.16'], 'geff_spec': ['pydantic>=2.11', 'annotated-types', 'pydantic-core', 'typing-inspection'], 'polars': ['polars-runtime-32'], 'dask': ['click', 'cloudpickle', 'fsspec', 'partd', 'pyyaml', 'toolz'], 'partd': ['locket'], 'blosc2': ['ndindex', 'msgpack', 'numexpr'], 'numcodecs': ['deprecated', 'msgpack', 'wrapt'], 'rich': ['markdown-it-py', 'pygments'], 'pydantic': ['annotated-types', 'pydantic-core', 'typing-extensions>=4.13', 'typing-inspection'], 'skimage': ['imageio', 'pillow', 'tifffile', 'lazy-loader', 'networkx']}
PIP_DEPENDENCIES = [spec for _, spec in PACKAGE_SPECS.values()]
REQUIRED_MODULES = {name: module for name, (module, _) in PACKAGE_SPECS.items() if module}
FALLBACK_ARTIFACT_SLUGS = ['biohub-tracking-support-pack-v1']
ALLOW_PIP_INSTALL = os.environ.get('BIOHUB_ALLOW_PIP_INSTALL', '0') != '0'

# Check whether a required python module is unavailable
def module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None

# Verify that a candidate directory contains the required repository and model weights
def has_model_artifact(path: Path) -> bool:
    has_repo_dir = (path / 'repo').exists()
    has_weights_dir = (path / 'weights' / METHOD / 'split_0' / 'edge_predictor_best.pth').exists()
    has_repo_zip = (path / 'repo.zip').exists()
    has_weights_zip = (path / 'weights.zip').exists()
    return has_repo_dir and has_weights_dir or (has_repo_zip and has_weights_zip)

# Load the artifact manifest when it is available
def artifact_manifest(path: Path) -> dict:
    manifest = path / 'ARTIFACT_MANIFEST.json'

    if not manifest.exists():
        return {}

    try:
        return json.loads(manifest.read_text())
    except Exception:
        return {}

# Check whether an artifact matches the requested model package
def artifact_matches_target(path: Path) -> bool:
    if ALLOW_ARTIFACT_FALLBACK:
        return True
    manifest = artifact_manifest(path)
    artifact_name = str(manifest.get('artifact_name', ''))
    path_text = str(path)
    return TARGET_ARTIFACT_SLUG in {artifact_name, path.name} or TARGET_ARTIFACT_SLUG in path_text

# Build the candidate artifact locations for a dataset slug
def candidate_roots_for_slug(slug: str) -> list[Path]:
    return [Path(f'/kaggle/input/datasets/pilkwang/{slug}'), Path(f'/kaggle/input/{slug}'), Path(f'/kaggle/input/{slug}/{slug}'), Path(f'PublicNotebook/{slug}')]

# Locate the primary biohub model artifact across supported kaggle paths
def find_artifacts_root() -> Path:
    candidates: list[Path] = []

    for env_name in ['BIOHUB_MODEL_ARTIFACTS', 'BIOHUB_ARTIFACTS']:
        explicit = os.environ.get(env_name, '').strip()

        if explicit:
            candidates.append(Path(explicit))
    candidates.append(PRIMARY_ARTIFACT_MANIFEST.parent)
    candidates.extend(candidate_roots_for_slug(TARGET_ARTIFACT_SLUG))

    if ALLOW_ARTIFACT_FALLBACK:
        for slug in FALLBACK_ARTIFACT_SLUGS:
            candidates.extend(candidate_roots_for_slug(slug))
    input_root = Path('/kaggle/input')

    if input_root.exists():
        for child in input_root.iterdir():
            if not child.is_dir():
                continue
            child_text = str(child)

            if TARGET_ARTIFACT_SLUG in child_text or ALLOW_ARTIFACT_FALLBACK:
                candidates.append(child)
                candidates.append(child / child.name)

                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.append(grandchild)
    seen: set[Path] = set()

    for candidate in candidates:
        candidate = candidate.expanduser()

        if candidate in seen:
            continue
        seen.add(candidate)

        if has_model_artifact(candidate) and artifact_matches_target(candidate):
            return candidate
    checked = '\n'.join((str(path) for path in candidates[:80]))
    raise FileNotFoundError(f'Could not find the required model artifact. Expected slug: {TARGET_ARTIFACT_SLUG}\nAttach the newly uploaded support dataset, or set BIOHUB_MODEL_ARTIFACTS.\nTo debug with an older artifact, set BIOHUB_ALLOW_ARTIFACT_FALLBACK = 1.\nChecked:\n' + checked)

# Check whether a directory contains installable offline package files
def _has_package_file(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    patterns = ('*.whl', '*.tar.gz', '*.zip')
    return any((any(path.glob(pattern)) for pattern in patterns))

# Collect directories that contain offline dependency packages
def find_offline_package_dirs(artifacts: Path) -> list[Path]:
    candidates: list[Path] = [artifacts / 'wheels', artifacts, Path('/kaggle/working'), Path('/kaggle/working/wheels')]
    input_root = Path('/kaggle/input')

    if input_root.exists():
        for child in input_root.iterdir():
            if child.is_dir():
                candidates.extend([child / 'wheels', child])

                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.extend([grandchild / 'wheels', grandchild])
    out: list[Path] = []
    seen: set[Path] = set()

    for candidate in candidates:
        candidate = candidate.expanduser()

        if candidate in seen:
            continue
        seen.add(candidate)

        if _has_package_file(candidate):
            out.append(candidate)
    return out

# Remove selected imported modules so refreshed packages can be reloaded cleanly
def purge_imported_modules(package_names: list[str]) -> None:
    roots = {'tracksdata'}

    for name in package_names:
        if name in PACKAGE_SPECS:
            module = PACKAGE_SPECS[name][0]
            roots.add(module.split('.')[0])

        if name == 'polars':
            roots.add('polars')

    for root in roots:
        for module_name in list(sys.modules):
            if module_name == root or module_name.startswith(root + '.'):
                sys.modules.pop(module_name, None)

# Verify that the installed polars runtime is compatible with the pipeline
def polars_runtime_ready() -> bool:
    try:
        import polars as _pl
        from polars._plr import PySeries as _PySeries
        _ = _PySeries
        return hasattr(_pl, 'Float16') and _pl.Series([-999999.0], dtype = _pl.Float64).dtype == _pl.Float64
    except Exception:
        return False

# Identify installed packages that need a compatible runtime refresh
def packages_requiring_refresh() -> list[str]:
    refresh: list[str] = []

    if not module_missing('polars') and (not polars_runtime_ready()):
        refresh.append('polars')

    if not module_missing('zarr'):
        try:
            import zarr as _zarr
            version_text = str(getattr(_zarr, '__version__', '0'))
            major = int(version_text.split('.', 1)[0])

            if major < 3:
                refresh.append('zarr')
        except Exception:
            refresh.append('zarr')
    return refresh

# Build the dependency specification list for missing or incompatible packages
def dependency_specs_for(missing: list[str]) -> list[str]:
    specs: list[str] = []
    seen: set[str] = set()

    # Add a dependency specification once while preserving order
    def add(spec: str) -> None:
        key = spec.lower()

        if key not in seen:
            seen.add(key)
            specs.append(spec)

    for name in missing:
        if name in PACKAGE_SPECS:
            add(PACKAGE_SPECS[name][1])

        for spec in EXTRA_SPECS_BY_NAME.get(name, []):
            add(spec)
    return specs

# Collect import errors for required runtime modules
def import_failures() -> dict[str, str]:
    failures: dict[str, str] = {}

    for name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[name] = f'{type(exc).__name__}: {exc}'
    return failures

# Map import failures back to package names that need installation
def missing_names_from_failures(failures: dict[str, str]) -> list[str]:
    names: list[str] = []
    module_to_name = {module: name for name, module in REQUIRED_MODULES.items()}

    for message in failures.values():
        match = re.search('No module named [\'\\"]([^\'\\"]+)[\'\\"]', message)

        if match:
            module = match.group(1).split('.')[0]
        else:
            match = re.search('module [\'\\"]([^\'\\"]+)[\'\\"] has no attribute', message)

            if not match:
                continue
            module = match.group(1).split('.')[0]
        name = module_to_name.get(module)

        if name and name not in names:
            names.append(name)
    return names

# Install missing dependencies from offline wheels or the allowed fallback source
def install_missing_dependencies(missing: list[str], artifacts: Path) -> None:
    specs = dependency_specs_for(missing)
    force_reinstall = bool({'polars', 'zarr'} & set(missing))

    if not specs:
        return
    package_dirs = find_offline_package_dirs(artifacts)

    if package_dirs:
        offline_cmd = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps']

        if force_reinstall:
            offline_cmd.append('--force-reinstall')

        for package_dir in package_dirs:
            offline_cmd.extend(['--find-links', str(package_dir)])
        offline_cmd.extend(specs)
        print('Installing missing packages from offline package dirs:', missing)
        print('Dependency resolver is disabled with --no-deps to avoid replacing Kaggle numpy/scipy in a live kernel.')
        print('Offline package dirs:', [str(path) for path in package_dirs])
        result = subprocess.run(offline_cmd, text = True, capture_output = True)

        if result.returncode == 0:
            purge_imported_modules(missing)
            print('Offline dependency install succeeded.')
            return
        print('Offline dependency install failed. Last pip output:')
        print((result.stdout or '')[-2000:])
        print((result.stderr or '')[-2000:])

    if ALLOW_PIP_INSTALL:
        online_cmd = [sys.executable, '-m', 'pip', 'install', '--no-deps']

        if force_reinstall:
            online_cmd.append('--force-reinstall')
        online_cmd.extend(specs)
        print('Installing missing packages from PyPI:', missing)
        result = subprocess.run(online_cmd, text = True, capture_output = True)

        if result.returncode == 0:
            purge_imported_modules(missing)
            print('PyPI dependency install succeeded.')
            return
        print('PyPI dependency install failed. Last pip output:')
        print((result.stdout or '')[-2000:])
        print((result.stderr or '')[-2000:])
    command = 'pip install tracksdata zarr>=3.0.10,<4 pyscipopt geff geff-spec ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy donfig numcodecs'
    raise ImportError('Missing required packages or dependency wheels: ' + ', '.join(missing) + '\nAttach the support dataset with offline wheels. If supplying Kaggle dependency input instead, use:\n' + command + '\nDo not quote zarr>=3.0.10,<4 in Kaggle dependency input.')

# Resolve and verify all runtime dependencies before inference starts
def ensure_dependencies(artifacts: Path) -> None:
    for _ in range(5):
        refresh = packages_requiring_refresh()

        if refresh:
            install_missing_dependencies(refresh, artifacts)
            continue
        missing = [pkg for pkg, module in REQUIRED_MODULES.items() if module_missing(module)]

        if missing:
            install_missing_dependencies(missing, artifacts)
            continue
        failures = import_failures()

        if not failures:
            print('Required graph/Zarr/ILP packages import successfully.')
            return
        missing_from_import = missing_names_from_failures(failures)

        if missing_from_import:
            install_missing_dependencies(missing_from_import, artifacts)
            continue
        raise ImportError('Required packages are present but failed to import. This may indicate a binary dependency mismatch in the live notebook kernel. Keep Kaggle dependency input empty and attach the wheels artifact.\n' + json.dumps(failures, indent = 2))
    failures = import_failures()
    raise ImportError('Dependency recovery did not converge after repeated offline installs. The attached support artifact may be missing wheels.\n' + json.dumps(failures, indent = 2))

# Remove an existing file, symlink, or directory before materialization
def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)

# Materialize a directory tree by copying a folder or extracting its archive
def copy_or_extract_tree(src_dir: Path, src_zip: Path, dst: Path) -> None:
    remove_path(dst)

    if src_dir.exists() and src_dir.is_dir():
        shutil.copytree(src_dir, dst)
        return

    if src_zip.exists() and src_zip.is_file():
        dst.mkdir(parents = True, exist_ok = True)

        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(dst)
        return
    raise FileNotFoundError(f'Missing source tree or zip: {src_dir} / {src_zip}')

# Reuse an existing directory through a symlink with a copy fallback
def link_or_copy_tree(src: Path, dst: Path) -> None:
    remove_path(dst)

    try:
        os.symlink(src, dst, target_is_directory = True)
    except Exception:
        shutil.copytree(src, dst)

# Prepare the inference repository and model weights inside the working directory
def materialize_inference_repo(artifacts: Path) -> None:
    predictions_src = REPO_DIR / 'predictions'

    if predictions_src.exists():
        if RESUME_PREDICTIONS_PATH.exists():
            remove_path(predictions_src)
        else:
            predictions_src.rename(RESUME_PREDICTIONS_PATH)
    copy_or_extract_tree(artifacts / 'repo', artifacts / 'repo.zip', REPO_DIR)
    weights_src = artifacts / 'weights'
    weights_zip = artifacts / 'weights.zip'
    weights_dst = REPO_DIR / 'weights'

    if weights_src.exists() and weights_src.is_dir():
        link_or_copy_tree(weights_src, weights_dst)
    elif weights_zip.exists() and weights_zip.is_file():
        remove_path(weights_dst)
        weights_dst.mkdir(parents = True, exist_ok = True)

        with zipfile.ZipFile(weights_zip) as zf:
            zf.extractall(weights_dst)
    else:
        raise FileNotFoundError(f'Missing weights tree or zip under {artifacts}')

    if RESUME_PREDICTIONS_PATH.exists():
        predictions_dst = REPO_DIR / 'predictions'
        remove_path(predictions_dst)
        RESUME_PREDICTIONS_PATH.rename(predictions_dst)
    required = [REPO_DIR / 'scripts' / 'predict_unet_transformer.py', REPO_DIR / WEIGHTS_RELATIVE]
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError('Materialized inference repo is incomplete:\n' + '\n'.join(missing))
    print('Inference repo:', REPO_DIR)
    print('Weights:', REPO_DIR / WEIGHTS_RELATIVE)

os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'] = '/kaggle/input/datasets/reyhanksatria/biohub-deepcenterunet3d-center-prior-v1/weights/full_frame_center/best.pt'

# Locate the primary artifact and materialize all required runtime dependencies
ARTIFACTS = find_artifacts_root()
print('ARTIFACTS:', ARTIFACTS)
print('Has offline wheels:', (ARTIFACTS / 'wheels').exists())
manifest_info = artifact_manifest(ARTIFACTS)

if manifest_info:
    print('Artifact name:', manifest_info.get('artifact_name'))
    print('Weight sha256:', manifest_info.get('model', {}).get('weight_sha256'))
    print('Weight path:', manifest_info.get('model', {}).get('weight_path'))
    _expected_primary_sha256 = '12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771'
    _actual_primary_sha256 = str(manifest_info.get('model', {}).get('weight_sha256', ''))

    if _actual_primary_sha256 != _expected_primary_sha256:
        raise RuntimeError(f"Primary model checksum mismatch: expected {_expected_primary_sha256}, got {_actual_primary_sha256 or 'missing'}")

ensure_dependencies(ARTIFACTS)
materialize_inference_repo(ARTIFACTS)

import hashlib as _integrity_hashlib

# Verify the support repository and model checkpoints before dynamic source patching
_support_expected_sha256 = {'scripts/augmentations.py': '13db09817bf492f8d0f710a0a4d09776320b262060167055090a303fc6057f4e', 'scripts/dataspec.py': 'e69bf952fb985477ac50ff8598a35020c95d20a035a09b81ab4056e655dd311f', 'scripts/evaluate.py': '614813cc51c3581c6ccda4bb20725a19da8ecac4a27620654bfca58319cffa3c', 'scripts/predict_unet_transformer.py': 'c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9', 'scripts/train_unet_transformer.py': 'c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea', 'src/biohub_tracking/__init__.py': '26a18d8da84e40da73281a48ebc3017d847a2e57431ab63e8629d2109e6e8571', 'src/biohub_tracking/division_metrics.py': 'd1cf1e0a43009d02174f1699ce2aa28458a2220ac4b521731d3bcf31cf8c76be', 'src/biohub_tracking/img_proc.py': '00e8ef0adc8b39f1aaaa547ea6197b906bf9e8c009e339d3e95f8f8dbf31be3f', 'src/biohub_tracking/io.py': 'efae135b088cecaab463d889f16c885ef6da3ad27b0747327d8ddc28d866b7bd', 'src/biohub_tracking/metrics.py': '31baf45b54c78f68bab4f65dd8f4b38bca702abb644171c6df7c46cdeef55d83', 'src/biohub_tracking/models/__init__.py': 'ab7587ef79856bae50d24b62e5805092d0459ee1c586522b763f9ef70c093e1d', 'src/biohub_tracking/models/simple_node_transformer.py': 'b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e', 'src/biohub_tracking/models/temporal_unet.py': 'd809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac'}
_support_expected_manifest_sha256 = '978b626d1fd1e7397435a437dfe68691defe1572fc3c20e61012d7c9b52ed029'
_primary_expected_sha256 = '12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771'
_deepcenter_expected_sha256 = '8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0'

# Compute a sha256 checksum for runtime integrity validation
def _integrity_sha256_file(path: Path) -> str:
    digest = _integrity_hashlib.sha256()

    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

_support_materialized_paths = {path.relative_to(REPO_DIR).as_posix(): path for path in REPO_DIR.rglob('*.py')}
_support_actual_names = set(_support_materialized_paths)
_support_expected_names = set(_support_expected_sha256)

if _support_actual_names != _support_expected_names:
    raise RuntimeError({'support_repo_python_files_missing': sorted(_support_expected_names - _support_actual_names), 'support_repo_python_files_extra': sorted(_support_actual_names - _support_expected_names)})

_support_actual_sha256 = {relative: _integrity_sha256_file(_support_materialized_paths[relative]) for relative in sorted(_support_materialized_paths)}

if _support_actual_sha256 != _support_expected_sha256:
    raise RuntimeError({'support_repo_python_checksum_mismatch': {relative: {'expected': _support_expected_sha256[relative], 'actual': _support_actual_sha256[relative]} for relative in sorted(_support_expected_sha256) if _support_actual_sha256[relative] != _support_expected_sha256[relative]}})

_support_manifest_bytes = ''.join((f'{_support_actual_sha256[relative]}  {relative}\n' for relative in sorted(_support_actual_sha256))).encode('utf-8')
_support_actual_manifest_sha256 = _integrity_hashlib.sha256(_support_manifest_bytes).hexdigest()

if _support_actual_manifest_sha256 != _support_expected_manifest_sha256:
    raise RuntimeError(f'Support repo manifest checksum mismatch: expected {_support_expected_manifest_sha256}, got {_support_actual_manifest_sha256}')

_primary_materialized_path = REPO_DIR / WEIGHTS_RELATIVE
_primary_actual_sha256 = _integrity_sha256_file(_primary_materialized_path)

if _primary_actual_sha256 != _primary_expected_sha256:
    raise RuntimeError(f'Materialized primary model checksum mismatch: expected {_primary_expected_sha256}, got {_primary_actual_sha256}')

_deepcenter_candidate_strings = [os.environ.get('BIOHUB_DEEPCENTER_CHECKPOINT', '').strip(), '/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt', '/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt']
_deepcenter_candidates = []

for _candidate_string in _deepcenter_candidate_strings:
    if not _candidate_string:
        continue
    _candidate_path = Path(_candidate_string)

    if _candidate_path not in _deepcenter_candidates:
        _deepcenter_candidates.append(_candidate_path)

_deepcenter_materialized_path = next((path for path in _deepcenter_candidates if path.is_file()), None)

if _deepcenter_materialized_path is None:
    raise FileNotFoundError({'missing_deepcenter_checkpoint': [str(path) for path in _deepcenter_candidates]})

_deepcenter_actual_sha256 = _integrity_sha256_file(_deepcenter_materialized_path)

if _deepcenter_actual_sha256 != _deepcenter_expected_sha256:
    raise RuntimeError(f'DeepCenter checkpoint checksum mismatch: expected {_deepcenter_expected_sha256}, got {_deepcenter_actual_sha256}')

os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'] = str(_deepcenter_materialized_path)
print('Support repo Python manifest SHA256:', _support_actual_manifest_sha256)
print('Primary materialized SHA256:', _primary_actual_sha256)
print('DeepCenter materialized SHA256:', _deepcenter_actual_sha256)

import hashlib as _hashlib

# Locate and verify the independent secondary model used for dual-seed inference
_secondary_manifest_explicit = Path(os.environ.get('BIOHUB_SECONDARY_ARTIFACT_MANIFEST', '/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json'))
_secondary_expected_sha256 = '9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f'
_secondary_slug = 'biohub-temporal-unet3d-seed314159-v1'

# Locate the independent secondary model artifact by its expected checksum
def _find_secondary_artifact_root() -> tuple[Path, dict]:
    candidates = [_secondary_manifest_explicit, Path(f'/kaggle/input/{_secondary_slug}/ARTIFACT_MANIFEST.json'), Path(f'/kaggle/input/datasets/pilkwang/{_secondary_slug}/ARTIFACT_MANIFEST.json')]
    input_root = Path('/kaggle/input')

    if input_root.exists():
        candidates.extend(input_root.rglob('ARTIFACT_MANIFEST.json'))
    seen = set()

    for manifest_path in candidates:
        manifest_path = manifest_path.expanduser()

        if manifest_path in seen or not manifest_path.is_file():
            continue
        seen.add(manifest_path)

        try:
            info = json.loads(manifest_path.read_text())
        except Exception:
            continue
        sha256 = str(info.get('model', {}).get('weight_sha256', ''))

        if sha256 == _secondary_expected_sha256:
            return (manifest_path.parent, info)
    raise FileNotFoundError('Could not find the independent-seed artifact with weight SHA256 ' + _secondary_expected_sha256)

SECONDARY_ARTIFACTS, secondary_manifest_info = _find_secondary_artifact_root()
SECONDARY_WEIGHTS_ROOT = WORKING_DIR / 'secondary_seed_weights'
copy_or_extract_tree(SECONDARY_ARTIFACTS / 'weights', SECONDARY_ARTIFACTS / 'weights.zip', SECONDARY_WEIGHTS_ROOT)
SECONDARY_WEIGHTS_PATH = SECONDARY_WEIGHTS_ROOT / 'unet_transformer' / 'split_0' / 'edge_predictor_best.pth'
SECONDARY_CONFIG_PATH = SECONDARY_WEIGHTS_PATH.parent / 'config.json'

for _required_secondary_path in (SECONDARY_WEIGHTS_PATH, SECONDARY_CONFIG_PATH):
    if not _required_secondary_path.is_file():
        raise FileNotFoundError(f'Missing secondary model file: {_required_secondary_path}')

# Compute a sha256 checksum for a model file
def _sha256_file(path: Path) -> str:
    digest = _hashlib.sha256()

    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

_secondary_actual_sha256 = _sha256_file(SECONDARY_WEIGHTS_PATH)

if _secondary_actual_sha256 != _secondary_expected_sha256:
    raise RuntimeError(f'Secondary model checksum mismatch: expected {_secondary_expected_sha256}, got {_secondary_actual_sha256}')

os.environ['BIOHUB_SECONDARY_WEIGHTS'] = str(SECONDARY_WEIGHTS_PATH)
os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'] = '0.15'
print('Secondary artifact:', SECONDARY_ARTIFACTS)
print('Secondary weight:', SECONDARY_WEIGHTS_PATH)
print('Secondary SHA256:', _secondary_actual_sha256)
print('Secondary edge-logit weight:', os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'])
print('Secondary edge-feature TTA blend:', os.environ['BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT'])
os.environ['BIOHUB_SECONDARY_DETECTION_WEIGHT'] = '0.80'
os.environ['BIOHUB_SECONDARY_LINK_MODE'] = 'low_margin_consensus'
os.environ['BIOHUB_SECONDARY_MIX_TEMPERATURE'] = '1'
os.environ['BIOHUB_SECONDARY_LOW_MARGIN_MAX'] = '0.35'
os.environ['BIOHUB_DUAL_SEED_EDGE_THRESHOLD'] = '0.48'
_runtime_integrity_receipt = {'status': 'complete_label_free_runtime_integrity', 'verified_before_dynamic_source_patch': True, 'support_repo_python_file_count': len(_support_actual_sha256), 'support_repo_python_sha256': _support_actual_sha256, 'support_repo_python_manifest_sha256': _support_actual_manifest_sha256, 'checkpoint_sha256': {'primary': _primary_actual_sha256, 'secondary': _secondary_actual_sha256, 'deepcenter': _deepcenter_actual_sha256}, 'materialized_paths': {'primary': str(_primary_materialized_path), 'secondary': str(SECONDARY_WEIGHTS_PATH), 'deepcenter': str(_deepcenter_materialized_path)}, 'ground_truth_accessed': False}
_runtime_integrity_receipt_path = WORKING_DIR / 'bidirectional_production_runtime_integrity.json'
_runtime_integrity_receipt_path.write_text(json.dumps(_runtime_integrity_receipt, indent = 2, sort_keys = True) + '\n', encoding = 'utf-8')
print('Runtime integrity receipt:', _runtime_integrity_receipt_path)

import torch as _torch

# Require a cuda device before launching gpu inference
if not _torch.cuda.is_available():
    raise RuntimeError('CUDA GPU is required for this notebook. Enable a Kaggle GPU accelerator and commit again.')

print('CUDA device:', _torch.cuda.get_device_name(0))

# Apply eight-view d4 detection tta before the association patches
_ps = REPO_DIR / 'scripts' / 'predict_unet_transformer.py'
_s = _ps.read_text()
_old = _exact_text('20202020202020206966206366672e6465745f7474613a0a2020202020202020202020207474615f666c697073203d205b282d312c292c20282d322c292c20282d322c202d31295d0a202020202020202020202020666f722064696d7320696e207474615f666c6970733a0a20202020202020202020202020202020696d67735f666c6970203d20696d67732e666c69702864696d73290a202020202020202020202020202020205f2c206465745f666c6970203d206d6f64656c2e656e636f646528696d67735f666c6970290a20202020202020202020202020202020666f72206620696e2072616e67652857293a0a20202020202020202020202020202020202020206465745f6c6f676974735b665d203d206465745f6c6f676974735b665d202b206465745f666c69705b665d2e666c69702864696d73290a2020202020202020202020202020202064656c20696d67735f666c69702c206465745f666c69700a202020202020202020202020666f72206620696e2072616e67652857293a0a202020202020202020202020202020206465745f6c6f676974735b665d203d206465745f6c6f676974735b665d202f2034')
_new = _patch_text("""# Expand detection TTA with flips rotations and transposed views
if cfg.det_tta:
    _nv = 1

    for dims in [(-1,), (-2,), (-2, -1)]:
        imgs_flip = imgs.flip(dims)
        _, det_flip = model.encode(imgs_flip)

        for f in range(W):
            det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
        del imgs_flip, det_flip
        _nv += 1

    for _k in (1, 3):
        imgs_rot = torch.rot90(imgs, _k, dims = (-2, -1))
        _, det_rot = model.encode(imgs_rot)

        for f in range(W):
            det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims = (-2, -1))
        del imgs_rot, det_rot
        _nv += 1

    imgs_t = imgs.transpose(-1, -2)
    _, det_t = model.encode(imgs_t)

    for f in range(W):
        det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
    del imgs_t, det_t
    _nv += 1

    imgs_at = torch.rot90(imgs, 1, dims = (-2, -1)).transpose(-1, -2)
    _, det_at = model.encode(imgs_at)

    for f in range(W):
        det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims = (-2, -1))
    del imgs_at, det_at
    _nv += 1

    for f in range(W):
        det_logits[f] = det_logits[f] / _nv
""", indent = 8, trailing_newline = False)

if _old in _s:
    _ps.write_text(_s.replace(_old, _new))
    print('Eight-view D4 detection TTA patch applied')
else:
    raise RuntimeError('Could not apply the eight-view D4 detection TTA patch')

_s = _ps.read_text()
_ensemble_replacements = [(_exact_text('20202020646f776e73616d706c653a207475706c655b696e742c202e2e2e5d203d2028312c20342c2034292c0a29202d3e207475706c655b6e702e6e6461727261792c206c6973745b7475706c655b696e742c20696e742c20666c6f61742c20666c6f61745d5d5d3a'), _patch_text("""# Extend the prediction function with secondary model parameters
    downsample: tuple[int, ...] = (1, 4, 4),
    secondary_model: UNetNodeTransformer | None = None,
    secondary_edge_weight: float = 0.0,
    secondary_detection_weight: float = 0.0,
    secondary_link_mode: str = "raw",
    secondary_mix_temperature: float = 1.0,
    secondary_low_margin_max: float = 0.2,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
""", indent = 0, trailing_newline = False)), (_exact_text('202020202020202020202020666f72206620696e2072616e67652857293a0a202020202020202020202020202020206465745f6c6f676974735b665d203d206465745f6c6f676974735b665d202f205f6e760a0a202020202020202064656c20696d6773'), _patch_text("""# Add secondary model detection and feature-ensemble blending
    for f in range(W):
        det_logits[f] = det_logits[f] / _nv

secondary_unet_out = None

if secondary_model is not None:
    secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)
    _secondary_edge_tta = os.environ.get('BIOHUB_SECONDARY_EDGE_FEATURE_TTA', '0') != '0'
    _secondary_unet_acc = secondary_unet_out.clone() if _secondary_edge_tta else None

    if secondary_detection_weight > 0.0:
        if cfg.det_tta:
            _secondary_nv = 1

            for dims in [(-1,), (-2,), (-2, -1)]:
                secondary_imgs_flip = imgs.flip(dims)
                _secondary_u_flip, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)

                for f in range(W):
                    secondary_det_logits[f] = secondary_det_logits[f] + secondary_det_flip[f].flip(dims)

                if _secondary_edge_tta:
                    _secondary_unet_acc = _secondary_unet_acc + _secondary_u_flip.flip(dims)
                del secondary_imgs_flip, secondary_det_flip, _secondary_u_flip
                _secondary_nv += 1

            for _k in (1, 3):
                secondary_imgs_rot = torch.rot90(imgs, _k, dims = (-2, -1))
                _secondary_u_rot, secondary_det_rot = secondary_model.encode(secondary_imgs_rot)

                for f in range(W):
                    secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(secondary_det_rot[f], -_k, dims = (-2, -1))

                if _secondary_edge_tta:
                    _secondary_unet_acc = _secondary_unet_acc + torch.rot90(_secondary_u_rot, -_k, dims = (-2, -1))
                del secondary_imgs_rot, secondary_det_rot, _secondary_u_rot
                _secondary_nv += 1
            secondary_imgs_t = imgs.transpose(-1, -2)
            _secondary_u_t, secondary_det_t = secondary_model.encode(secondary_imgs_t)

            for f in range(W):
                secondary_det_logits[f] = secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)

            if _secondary_edge_tta:
                _secondary_unet_acc = _secondary_unet_acc + _secondary_u_t.transpose(-1, -2)
            del secondary_imgs_t, secondary_det_t, _secondary_u_t
            _secondary_nv += 1
            secondary_imgs_at = torch.rot90(imgs, 1, dims = (-2, -1)).transpose(-1, -2)
            _secondary_u_at, secondary_det_at = secondary_model.encode(secondary_imgs_at)

            for f in range(W):
                secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(secondary_det_at[f].transpose(-1, -2), -1, dims = (-2, -1))

            if _secondary_edge_tta:
                _secondary_unet_acc = _secondary_unet_acc + torch.rot90(_secondary_u_at.transpose(-1, -2), -1, dims = (-2, -1))
            del secondary_imgs_at, secondary_det_at, _secondary_u_at
            _secondary_nv += 1

            for f in range(W):
                secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv

            if _secondary_edge_tta:
                if _secondary_unet_acc.shape != secondary_unet_out.shape:
                    raise RuntimeError(f'Secondary edge-feature TTA shape mismatch: {tuple(_secondary_unet_acc.shape)} vs {tuple(secondary_unet_out.shape)}')
                _secondary_delta = float((_secondary_unet_acc / _secondary_nv - secondary_unet_out).abs().mean())

                if _secondary_delta == 0.0:
                    raise RuntimeError('Secondary edge-feature TTA produced no feature change')
                _secondary_edge_tta_weight = float(os.environ.get('BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT', '1.0'))

                if not 0.0 < _secondary_edge_tta_weight <= 1.0:
                    raise RuntimeError(f'Secondary edge-feature TTA weight must be in (0, 1], got {_secondary_edge_tta_weight}')
                _secondary_tta_mean = _secondary_unet_acc / _secondary_nv
                secondary_unet_out = (1.0 - _secondary_edge_tta_weight) * secondary_unet_out + _secondary_edge_tta_weight * _secondary_tta_mean
                print('SECONDARY_EDGE_TTA_ACTIVE views =', _secondary_nv, 'weight =', _secondary_edge_tta_weight, 'mean_abs_feat_delta =', round(_secondary_delta, 6), flush = True)
                del _secondary_unet_acc

        for f in range(W):
            primary_det = det_logits[f]
            secondary_det = secondary_det_logits[f]
            primary_mean = primary_det.mean()
            secondary_mean = secondary_det.mean()
            primary_scale = primary_det.float().std(unbiased = False).clamp_min(1e-4)
            secondary_scale = secondary_det.float().std(unbiased = False).clamp_min(1e-4)
            scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
            secondary_det_aligned = (secondary_det - secondary_mean) * scale_ratio + primary_mean
            det_logits[f] = (1.0 - secondary_detection_weight) * primary_det + secondary_detection_weight * secondary_det_aligned

    del secondary_det_logits

del imgs
""", indent = 8, trailing_newline = False)), (_exact_text('202020202020202020202020656467655f6c6f676974735f70616972203d206d6f64656c2e707265646963745f6564676573280a20202020202020202020202020202020756e65745f666561745f7372632c20756e65745f666561745f7467742c0a20202020202020202020202020202020705f636f6f7264735f737263202a2064735f6172725f742c20705f636f6f7264735f746774202a2064735f6172725f742c0a20202020202020202020202020202020705f706f735f7372632c20705f706f735f7467742c0a20202020202020202020202020202020705f6d61736b5f7372632c20705f6d61736b5f7467742c0a202020202020202020202020292020232028312c206e5f7372632c206e5f746774290a0a202020202020202020202020726177203d20656467655f6c6f676974735f706169725b305d'), _patch_text("""# Pass secondary model settings into the prediction call
edge_logits_pair = model.predict_edges(unet_feat_src, unet_feat_tgt, p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t, p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt)

if secondary_model is not None:
    if secondary_unet_out is None:
        raise RuntimeError('Secondary model is loaded but its feature map is missing')
    secondary_feat_src = secondary_model._index_features(secondary_unet_out[:, f_idx], p_coords_src, p_mask_src)
    secondary_feat_tgt = secondary_model._index_features(secondary_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt)
    secondary_logits_pair = secondary_model.predict_edges(secondary_feat_src, secondary_feat_tgt, p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t, p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt)

    if secondary_link_mode == 'raw':
        secondary_for_mix = secondary_logits_pair
        blend_weight = secondary_edge_weight
    elif secondary_link_mode in {'calibrated', 'adaptive', 'low_margin_consensus'}:
        primary_center = edge_logits_pair.mean(dim = 1, keepdim = True)
        primary_scale = edge_logits_pair.float().std(dim = 1, keepdim = True, unbiased = False).clamp_min(0.0001)
        secondary_center = secondary_logits_pair.mean(dim = 1, keepdim = True)
        secondary_scale = secondary_logits_pair.float().std(dim = 1, keepdim = True, unbiased = False).clamp_min(0.0001)
        secondary_scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
        secondary_for_mix = (secondary_logits_pair - secondary_center) * secondary_scale_ratio + primary_center

        if secondary_link_mode == 'calibrated':
            blend_weight = secondary_edge_weight
        elif secondary_link_mode == 'adaptive':
            if n_src >= 2:
                primary_probs = torch.softmax(edge_logits_pair[0], dim = 0)
                secondary_probs = torch.softmax(secondary_for_mix[0], dim = 0)
                primary_top2 = torch.topk(primary_probs, k = 2, dim = 0)
                secondary_top2 = torch.topk(secondary_probs, k = 2, dim = 0)
                primary_margin = primary_top2.values[0] - primary_top2.values[1]
                secondary_margin = secondary_top2.values[0] - secondary_top2.values[1]
                local_weight = (secondary_edge_weight + secondary_margin - primary_margin).clamp(0.15, 0.75)
                same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
                local_weight = torch.where(same_parent, torch.maximum(local_weight, torch.full_like(local_weight, secondary_edge_weight)), local_weight)
                blend_weight = local_weight.view(1, 1, -1)
            else:
                blend_weight = secondary_edge_weight
        elif n_src >= 2:
            primary_probs = torch.softmax(edge_logits_pair[0], dim = 0)
            secondary_probs = torch.softmax(secondary_for_mix[0], dim = 0)
            primary_top2 = torch.topk(primary_probs, k = 2, dim = 0)
            secondary_top2 = torch.topk(secondary_probs, k = 2, dim = 0)
            primary_margin = primary_top2.values[0] - primary_top2.values[1]
            same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
            uncertainty = ((secondary_low_margin_max - primary_margin) / secondary_low_margin_max).clamp(0.0, 1.0)
            local_weight = secondary_edge_weight * uncertainty
            local_weight = torch.where(same_parent, local_weight, torch.zeros_like(local_weight))
            blend_weight = local_weight.view(1, 1, -1)
        else:
            blend_weight = 0.0
    else:
        raise ValueError(f'Unsupported secondary link mode: {secondary_link_mode}')
    edge_logits_pair = (1.0 - blend_weight) * edge_logits_pair + blend_weight * secondary_for_mix

    if secondary_mix_temperature != 1.0:
        mixed_center = edge_logits_pair.mean(dim = 1, keepdim = True)
        edge_logits_pair = mixed_center + (edge_logits_pair - mixed_center) / secondary_mix_temperature

raw = edge_logits_pair[0]
""", indent = 12, trailing_newline = False)), (_exact_text('202020202020202064656c20756e65745f6f75740a'), _patch_text("""# Load and configure the independent secondary tracking model
del unet_out

if secondary_unet_out is not None:
    del secondary_unet_out
""", indent = 8, trailing_newline = True)), (_exact_text('202020206d6f64656c2c2077696e646f775f73697a652c20646f776e73616d706c65203d206c6f61645f6d6f64656c28776569676874735f706174682c20646576696365290a202020207072696e7428'), _patch_text("""# Fuse primary and secondary edge logits under the configured consensus rule
model, window_size, downsample = load_model(weights_path, device)

secondary_model = None
secondary_weights_text = os.environ.get("BIOHUB_SECONDARY_WEIGHTS", "").strip()
secondary_edge_weight = float(os.environ.get("BIOHUB_SECONDARY_EDGE_WEIGHT", "0"))
secondary_detection_weight = float(os.environ.get("BIOHUB_SECONDARY_DETECTION_WEIGHT", "0"))
secondary_link_mode = os.environ.get("BIOHUB_SECONDARY_LINK_MODE", "raw").strip()
secondary_mix_temperature = float(os.environ.get("BIOHUB_SECONDARY_MIX_TEMPERATURE", "1"))
secondary_low_margin_max = float(os.environ.get("BIOHUB_SECONDARY_LOW_MARGIN_MAX", "0.2"))
edge_candidate_threshold = float(os.environ.get("BIOHUB_DUAL_SEED_EDGE_THRESHOLD", str(cfg.threshold)))

if secondary_weights_text:
    if not 0.0 < secondary_edge_weight < 1.0:
        raise ValueError("BIOHUB_SECONDARY_EDGE_WEIGHT must be strictly between 0 and 1")

    if not 0.0 <= secondary_detection_weight < 1.0:
        raise ValueError("BIOHUB_SECONDARY_DETECTION_WEIGHT must be in the half-open interval [0, 1)")

    if secondary_link_mode not in { "raw", "calibrated", "adaptive", "low_margin_consensus" }:
        raise ValueError("BIOHUB_SECONDARY_LINK_MODE must be raw, calibrated, adaptive, " "or low_margin_consensus")

    if not 0.5 <= secondary_mix_temperature <= 2.0:
        raise ValueError("BIOHUB_SECONDARY_MIX_TEMPERATURE must be in [0.5, 2.0]")

    if not 0.0 < edge_candidate_threshold < 1.0:
        raise ValueError("BIOHUB_DUAL_SEED_EDGE_THRESHOLD must be strictly between 0 and 1")

    if not 0.0 < secondary_low_margin_max <= 1.0:
        raise ValueError("BIOHUB_SECONDARY_LOW_MARGIN_MAX must be in (0, 1]")
    secondary_model, secondary_window_size, secondary_downsample = load_model(Path(secondary_weights_text), device,)

    if secondary_window_size != window_size or secondary_downsample != downsample:
        raise ValueError("Primary and secondary models have incompatible inference grids: " f"primary: (window = {window_size}, downsample = {downsample}), " f"secondary: (window = {secondary_window_size}, downsample = {secondary_downsample})")
    cfg.threshold = edge_candidate_threshold
    print(f"Secondary model: {secondary_weights_text} | " f"edge weight = {secondary_edge_weight:.3f} | " f"detection weight = {secondary_detection_weight:.3f} | " f"link mode = {secondary_link_mode} | " f"temperature = {secondary_mix_temperature:.3f} | " f"low-margin max = {secondary_low_margin_max:.3f} | " f"edge threshold = {cfg.threshold:.3f}", flush = True,)

print(""", indent = 4, trailing_newline = False)), (_exact_text('20202020202020202020202020202020756e65745f62617463685f73697a653d756e65745f62617463685f73697a652c0a20202020202020202020202020202020646f776e73616d706c653d646f776e73616d706c652c0a20202020202020202020202029'), _patch_text("""# Add bidirectional association parameters to the prediction function
    unet_batch_size = unet_batch_size,
    downsample = downsample,
    secondary_model = secondary_model,
    secondary_edge_weight = secondary_edge_weight,
    secondary_detection_weight = secondary_detection_weight,
    secondary_link_mode = secondary_link_mode,
    secondary_mix_temperature = secondary_mix_temperature,
    secondary_low_margin_max = secondary_low_margin_max)
""", indent = 12, trailing_newline = False))]

for _patch_index, (_ensemble_old, _ensemble_new) in enumerate(_ensemble_replacements, start = 1):
    _ensemble_count = _s.count(_ensemble_old)

    if _ensemble_count != 1:
        raise RuntimeError(f'Calibrated dual-seed patch {_patch_index} expected one match, found {_ensemble_count}')
    _s = _s.replace(_ensemble_old, _ensemble_new, 1)

compile(_s, str(_ps), 'exec')
_ps.write_text(_s)
print('Calibrated dual-seed runtime patch applied')
os.environ['BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION'] = '0.90'

_s = _ps.read_text()
_guard_old = _exact_text('20202020202020202020202020202020202020206465745f6c6f676974735b665d203d2028312e30202d207365636f6e646172795f646574656374696f6e5f77656967687429202a207072696d6172795f646574202b207365636f6e646172795f646574656374696f6e5f776569676874202a207365636f6e646172795f6465745f616c69676e6564')
_guard_new = _patch_text("""# Compute reverse-time association logits for mutual temporal support
blended_det = (1.0 - secondary_detection_weight) * primary_det + secondary_detection_weight * secondary_det_aligned
primary_candidates = len(_detect_cells_pooled(primary_det[0], int(frame_indices[f]), cfg.det_threshold, pool_k))
blended_candidates = len(_detect_cells_pooled(blended_det[0], int(frame_indices[f]), cfg.det_threshold, pool_k))
minimum_retention = float(os.environ.get('BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION', '0.90'))
candidate_retention = blended_candidates / primary_candidates if primary_candidates else 1.0
use_primary_detection = bool(primary_candidates > 0 and candidate_retention < minimum_retention)
det_logits[f] = primary_det if use_primary_detection else blended_det

if int(frame_indices[f]) not in seen_frames:
    shard = os.environ.get('BIOHUB_GPU_SHARD', 'single').replace('/', '_')
    guard_log = Path('/kaggle/working') / f'retention_guard_{shard}.jsonl'
    guard_record = {'dataset': ds_path.stem, 'frame': int(frame_indices[f]), 'primary_candidates': int(primary_candidates), 'blended_candidates': int(blended_candidates), 'retention': float(candidate_retention), 'minimum_retention': float(minimum_retention), 'use_primary': bool(use_primary_detection)}

    with guard_log.open('a') as guard_handle:
        guard_handle.write(json.dumps(guard_record, sort_keys = True) + '\\n')

    if use_primary_detection:
        print('BIOHUB_RETENTION_GUARD ' + json.dumps(guard_record, sort_keys = True), flush = True)
""", indent = 20, trailing_newline = False)
_guard_matches = _s.count(_guard_old)

if _guard_matches != 1:
    raise RuntimeError(f'Retention guard expected one blend block, found {_guard_matches}')

_s = _s.replace(_guard_old, _guard_new, 1)
compile(_s, str(_ps), 'exec')
_ps.write_text(_s)
print('Frozen frame retention guard applied at ' + os.environ['BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION'])

import math as _bidirectional_math

_bidirectional_weight_guard = float(os.environ.get('BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT', '0'))

if not _bidirectional_math.isclose(_bidirectional_weight_guard, 0.15, rel_tol = 0.0, abs_tol = 1e-12):
    raise ValueError({'expected_bidirectional_weight': 0.15, 'actual_bidirectional_weight': _bidirectional_weight_guard})

_s = _ps.read_text()
_bi_old = _exact_text('202020202020202020202020656467655f6c6f676974735f70616972203d206d6f64656c2e707265646963745f656467657328756e65745f666561745f7372632c20756e65745f666561745f7467742c20705f636f6f7264735f737263202a2064735f6172725f742c20705f636f6f7264735f746774202a2064735f6172725f742c20705f706f735f7372632c20705f706f735f7467742c20705f6d61736b5f7372632c20705f6d61736b5f746774290a0a2020202020202020202020206966207365636f6e646172795f6d6f64656c206973206e6f74204e6f6e653a0a')
_bi_new = _patch_text("""# Fuse bidirectional association with harmonic probability support
edge_logits_pair = model.predict_edges(unet_feat_src, unet_feat_tgt, p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t, p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt,)

_bidirectional_weight = float(os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0"))

if _bidirectional_weight > 0.0:
    reverse_logits_native = model.predict_edges(unet_feat_tgt, unet_feat_src, p_coords_tgt * ds_arr_t, p_coords_src * ds_arr_t, p_pos_tgt, p_pos_src, p_mask_tgt, p_mask_src,)
    reverse_logits_pair = reverse_logits_native.transpose(1, 2)

    forward_center = edge_logits_pair.mean(dim = 1, keepdim = True)
    forward_scale = edge_logits_pair.float().std(dim = 1, keepdim = True, unbiased = False).clamp_min(1e-4)
    reverse_center = reverse_logits_pair.mean(dim = 1, keepdim = True)
    reverse_scale = reverse_logits_pair.float().std(dim = 1, keepdim = True, unbiased = False).clamp_min(1e-4)
    reverse_scale_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0)
    reverse_scale_ratio = reverse_scale_ratio.to(reverse_logits_pair.dtype)
    reverse_aligned = (reverse_logits_pair - reverse_center) * reverse_scale_ratio + forward_center
    forward_prob = torch.softmax(edge_logits_pair.float(), dim = 1).clamp_min(1e-8)
    reverse_prob = torch.softmax(reverse_aligned.float(), dim = 1).clamp_min(1e-8)
    harmonic_prob = 1.0 / ((1.0 - _bidirectional_weight) / forward_prob + _bidirectional_weight / reverse_prob)
    harmonic_prob = harmonic_prob / harmonic_prob.sum(dim = 1, keepdim = True).clamp_min(1e-8)
    harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))
    harmonic_center = harmonic_logits.mean(dim = 1, keepdim = True)
    harmonic_scale = harmonic_logits.std(dim = 1, keepdim = True, unbiased = False).clamp_min(1e-4)
    harmonic_scale_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)
    edge_logits_pair = ((harmonic_logits - harmonic_center) * harmonic_scale_ratio + forward_center).to(reverse_aligned.dtype)
    del (reverse_logits_native, reverse_logits_pair, reverse_aligned, forward_prob, reverse_prob, harmonic_prob, harmonic_logits,)

if secondary_model is not None:
""", indent = 12, trailing_newline = True)
_bi_count = _s.count(_bi_old)

if _bi_count != 1:
    raise RuntimeError(f'Bidirectional edge patch expected one transformed block, found {_bi_count}')

_s = _s.replace(_bi_old, _bi_new, 1)
_coordinate_manifest_old = _exact_text('20202020636f6f726473203d20636f6f7264732e617374797065286e702e696e743136290a2020202072657475726e20636f6f7264732c20616c6c5f6564676573')
_coordinate_manifest_new = _patch_text("""# Pass bidirectional association settings into the final prediction call
coords = coords.astype(np.int16)
_coordinate_manifest_arm = os.environ.get('BIOHUB_DIAGNOSTIC_ARM', '').strip()

if _coordinate_manifest_arm:
    import hashlib as _coordinate_hashlib
    _coordinate_shard = os.environ.get('BIOHUB_GPU_SHARD', 'single').replace('/', '_')
    _coordinate_array = np.ascontiguousarray(coords.astype('<i2', copy = False))
    _coordinate_frame_counts = [[int(_coordinate_t), int((_coordinate_array[:, 0] == _coordinate_t).sum())] for _coordinate_t in np.unique(_coordinate_array[:, 0])]
    _coordinate_record = {'columns': ['t', 'z', 'y', 'x'], 'coordinate_sha256': _coordinate_hashlib.sha256(_coordinate_array.tobytes(order = 'C')).hexdigest(), 'dataset': ds_path.stem, 'dtype': '<i2', 'frame_counts': _coordinate_frame_counts, 'rows': int(len(_coordinate_array)), 'stage': 'post_detection_pre_graph_pre_ilp'}
    _coordinate_manifest_path = Path('/kaggle/working') / f'detector_coordinates_{_coordinate_manifest_arm}_{_coordinate_shard}.jsonl'

    with _coordinate_manifest_path.open('a') as _coordinate_handle:
        _coordinate_handle.write(json.dumps(_coordinate_record, sort_keys = True) + '\\n')

return (coords, all_edges)
""", indent = 4, trailing_newline = False)
_coordinate_manifest_count = _s.count(_coordinate_manifest_old)

if _coordinate_manifest_count != 1:
    raise RuntimeError(f'Coordinate-manifest patch expected one pre-return block, found {_coordinate_manifest_count}')

_s = _s.replace(_coordinate_manifest_old, _coordinate_manifest_new, 1)
compile(_s, str(_ps), 'exec')
_ps.write_text(_s)
print('Bidirectional harmonic-probability association fusion applied | weight:', _bidirectional_weight_guard)
print('Pre-ILP detector-coordinate manifest hook applied')

# Extend the existing eight-view d4 ensemble into unet association features
_et_s = _ps.read_text()
_et_old = _patch_text("""# Expand detection TTA with flips rotations and transposed views
if cfg.det_tta:
    _nv = 1

    for dims in [(-1,), (-2,), (-2, -1)]:
        imgs_flip = imgs.flip(dims)
        _, det_flip = model.encode(imgs_flip)

        for f in range(W):
            det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
        del imgs_flip, det_flip
        _nv += 1

    for _k in (1, 3):
        imgs_rot = torch.rot90(imgs, _k, dims = (-2, -1))
        _, det_rot = model.encode(imgs_rot)

        for f in range(W):
            det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims = (-2, -1))
        del imgs_rot, det_rot
        _nv += 1

    imgs_t = imgs.transpose(-1, -2)
    _, det_t = model.encode(imgs_t)

    for f in range(W):
        det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
    del imgs_t, det_t
    _nv += 1

    imgs_at = torch.rot90(imgs, 1, dims = (-2, -1)).transpose(-1, -2)
    _, det_at = model.encode(imgs_at)

    for f in range(W):
        det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims = (-2, -1))
    del imgs_at, det_at
    _nv += 1

    for f in range(W):
        det_logits[f] = det_logits[f] / _nv
""", indent = 8, trailing_newline = False)
_et_new = _patch_text("""# Average detection logits and association features across eight D4 views
if cfg.det_tta:
    _edge_tta = os.environ.get('BIOHUB_EDGE_FEATURE_TTA', '0') != '0'
    _unet_acc = unet_out.clone() if _edge_tta else None
    _nv = 1

    for dims in [(-1,), (-2,), (-2, -1)]:
        imgs_flip = imgs.flip(dims)
        _u_flip, det_flip = model.encode(imgs_flip)

        for f in range(W):
            det_logits[f] = det_logits[f] + det_flip[f].flip(dims)

        if _edge_tta:
            _unet_acc = _unet_acc + _u_flip.flip(dims)
        del imgs_flip, det_flip, _u_flip
        _nv += 1

    for _k in (1, 3):
        imgs_rot = torch.rot90(imgs, _k, dims = (-2, -1))
        _u_rot, det_rot = model.encode(imgs_rot)

        for f in range(W):
            det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims = (-2, -1))

        if _edge_tta:
            _unet_acc = _unet_acc + torch.rot90(_u_rot, -_k, dims = (-2, -1))
        del imgs_rot, det_rot, _u_rot
        _nv += 1

    imgs_t = imgs.transpose(-1, -2)
    _u_t, det_t = model.encode(imgs_t)

    for f in range(W):
        det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)

    if _edge_tta:
        _unet_acc = _unet_acc + _u_t.transpose(-1, -2)
    del imgs_t, det_t, _u_t
    _nv += 1

    imgs_at = torch.rot90(imgs, 1, dims = (-2, -1)).transpose(-1, -2)
    _u_at, det_at = model.encode(imgs_at)

    for f in range(W):
        det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims = (-2, -1))

    if _edge_tta:
        _unet_acc = _unet_acc + torch.rot90(_u_at.transpose(-1, -2), -1, dims = (-2, -1))
    del imgs_at, det_at, _u_at
    _nv += 1

    for f in range(W):
        det_logits[f] = det_logits[f] / _nv

    if _edge_tta:
        if _unet_acc.shape != unet_out.shape:
            raise RuntimeError(f'Edge-feature TTA shape mismatch: {tuple(_unet_acc.shape)} vs {tuple(unet_out.shape)}')
        _delta = float((_unet_acc / _nv - unet_out).abs().mean())

        if _delta == 0.0:
            raise RuntimeError('Edge-feature TTA produced no feature change')
        unet_out = _unet_acc / _nv
        print('EDGE_TTA_ACTIVE views =', _nv, 'mean_abs_feat_delta =', round(_delta, 6), flush = True)
        del _unet_acc
""", indent = 8, trailing_newline = False)
_et_matches = _et_s.count(_et_old)

if _et_matches != 1:
    raise RuntimeError(f'Edge-feature TTA anchor expected one match, found {_et_matches}')
_et_s = _et_s.replace(_et_old, _et_new, 1)
compile(_et_s, str(_ps), 'exec')
_ps.write_text(_et_s)

if 'EDGE_TTA_ACTIVE' not in _ps.read_text():
    raise RuntimeError('Edge-feature TTA patch did not persist')
print('Edge-feature TTA patch installed and enabled')

# List all test movie identifiers available for inference
def list_test_stems() -> list[str]:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f'Test directory does not exist: {TEST_DIR}')
    stems = sorted((path.name[:-5] for path in TEST_DIR.iterdir() if path.name.endswith('.zarr')))

    if not stems:
        raise FileNotFoundError(f'No test .zarr files found in {TEST_DIR}')
    return stems

# Collect test movies and prepare deterministic prediction sharding
test_stems = list_test_stems()
print(f'Found {len(test_stems)} test videos')
print(test_stems[:10])
splits_path = REPO_DIR / 'kaggle_test_splits_50ep.json'
splits_path.parent.mkdir(parents = True, exist_ok = True)
splits_path.write_text(json.dumps([{'split': 0, 'train': [], 'test': test_stems}], indent = 2))
predict_cmd = [sys.executable, 'scripts/predict_unet_transformer.py', '--data-dir', str(TEST_DIR), '--splits', str(splits_path.name), '--split', '0', '--weights', WEIGHTS_RELATIVE, '--unet-batch-size', str(UNET_BATCH_SIZE), '--det-threshold', str(DET_THRESHOLD), '--ilp-edge-weight', str(ILP_EDGE_WEIGHT), '--ilp-appearance-weight', str(ILP_APPEARANCE_WEIGHT), '--ilp-disappearance-weight', str(ILP_DISAPPEARANCE_WEIGHT), '--ilp-division-weight', str(ILP_DIVISION_WEIGHT)]

if USE_ILP:
    predict_cmd.append('--use-ilp')

if SLICE:
    predict_cmd.extend(['--slice', SLICE])

# Read the cuda devices exposed to the current process
def _visible_cuda_tokens(count: int) -> list[str]:
    raw = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()

    if raw and raw != '-1':
        tokens = [token.strip() for token in raw.split(',') if token.strip()]

        if len(tokens) < count:
            raise RuntimeError(f'torch reports {count} CUDA devices but CUDA_VISIBLE_DEVICES = {raw!r}')
        return tokens[:count]
    return [str(index) for index in range(count)]

# Resolve the prediction output directory for the selected tracking method
def _prediction_dir_for_method(method: str) -> Path:
    matches = sorted((REPO_DIR / 'predictions').glob(f'*/{method}/split_0'))

    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one prediction directory for {method!r}, found {matches}')
    return matches[0]

# Wait until all expected prediction shards have been written
def _wait_for_prediction_shards(processes: dict[int, subprocess.Popen], commands: dict[int, list[str]]) -> None:
    while processes:
        failed: tuple[int, int] | None = None

        for shard_index, process in list(processes.items()):
            return_code = process.poll()

            if return_code is None:
                continue
            _join_streamed_process(process)
            del processes[shard_index]

            if return_code != 0:
                failed = (shard_index, return_code)
                break

        if failed is None:
            if processes:
                time.sleep(1.0)
            continue
        failed_index, failed_code = failed

        for process in processes.values():
            if process.poll() is None:
                process.terminate()

        for process in processes.values():
            try:
                process.wait(timeout = 30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _join_streamed_process(process)
        raise subprocess.CalledProcessError(failed_code, commands[failed_index])

# Merge per-device prediction shards into one complete prediction set
def _merge_prediction_shards(worker_count: int) -> Path:
    shard_dirs: list[Path] = []
    seen: set[str] = set()
    expected_all = set(test_stems)

    for shard_index in range(worker_count):
        shard_method = f'{METHOD}_gpu{shard_index}'
        shard_dir = _prediction_dir_for_method(shard_method)
        expected = set(test_stems[shard_index::worker_count])
        shard_paths = sorted(shard_dir.glob('*.geff'))
        found = {path.stem for path in shard_paths}

        if found != expected:
            raise RuntimeError(f'GPU shard {shard_index} output mismatch: missing = {sorted(expected - found)}, extra = {sorted(found - expected)}')
        overlap = seen & found

        if overlap:
            raise RuntimeError(f'Duplicate datasets across GPU shards: {sorted(overlap)}')
        seen.update(found)
        shard_dirs.append(shard_dir)

    if seen != expected_all:
        raise RuntimeError(f'Merged GPU shards do not cover the test set: missing = {sorted(expected_all - seen)}, extra = {sorted(seen - expected_all)}')
    username_roots = {shard_dir.parents[1] for shard_dir in shard_dirs}

    if len(username_roots) != 1:
        raise RuntimeError(f'GPU shards used inconsistent prediction roots: {username_roots}')
    import shutil as _shutil
    final_root = next(iter(username_roots)) / METHOD
    final_dir = final_root / 'split_0'
    staging_dir = final_root / 'split_0_dual_gpu_staging'

    if staging_dir.exists():
        if staging_dir.is_dir():
            _shutil.rmtree(staging_dir)
        else:
            staging_dir.unlink()
    staging_dir.mkdir(parents = True, exist_ok = False)

    for shard_dir in shard_dirs:
        for source in sorted(shard_dir.glob('*.geff')):
            destination = staging_dir / source.name

            if destination.exists():
                raise RuntimeError(f'Refusing to overwrite duplicate merged output: {destination}')
            _shutil.move(str(source), str(destination))
    merged = {path.stem for path in staging_dir.glob('*.geff')}

    if merged != expected_all:
        raise RuntimeError(f'Staged prediction directory failed verification: missing = {sorted(expected_all - merged)}, extra = {sorted(merged - expected_all)}')

    if final_dir.exists():
        if final_dir.is_dir():
            _shutil.rmtree(final_dir)
        else:
            final_dir.unlink()
    staging_dir.rename(final_dir)

    for shard_dir in shard_dirs:
        _shutil.rmtree(shard_dir.parent)
    print(f'Merged {len(merged)} prediction graphs into {final_dir}')
    return final_dir

_inference_resume_env_keys = ['BIOHUB_DET_THRESHOLD', 'BIOHUB_SECONDARY_EDGE_WEIGHT', 'BIOHUB_SECONDARY_DETECTION_WEIGHT', 'BIOHUB_SECONDARY_LINK_MODE', 'BIOHUB_SECONDARY_MIX_TEMPERATURE', 'BIOHUB_SECONDARY_LOW_MARGIN_MAX', 'BIOHUB_DUAL_SEED_EDGE_THRESHOLD', 'BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION', 'BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT', 'BIOHUB_BIDIRECTIONAL_FUSION_MODE', 'BIOHUB_EDGE_FEATURE_TTA', 'BIOHUB_SECONDARY_EDGE_FEATURE_TTA', 'BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT']
_test_prediction_signature = _resume_signature({'resume_schema': RESUME_SCHEMA_VERSION, 'experiment_tag': EXPERIMENT_TAG, 'method': METHOD, 'weights_relative': WEIGHTS_RELATIVE, 'primary_sha256': _primary_actual_sha256, 'secondary_sha256': _secondary_actual_sha256, 'test_stems': test_stems, 'unet_batch_size': UNET_BATCH_SIZE, 'use_ilp': USE_ILP, 'ilp_edge_weight': ILP_EDGE_WEIGHT, 'ilp_appearance_weight': ILP_APPEARANCE_WEIGHT, 'ilp_disappearance_weight': ILP_DISAPPEARANCE_WEIGHT, 'ilp_division_weight': ILP_DIVISION_WEIGHT, 'slice': SLICE, 'environment': {key: os.environ.get(key) for key in _inference_resume_env_keys}})
_test_prediction_state = _read_resume_json(TEST_PREDICTION_STATE_PATH)
_test_prediction_ready = bool(_test_prediction_state and _test_prediction_state.get('resume_schema') == RESUME_SCHEMA_VERSION and _test_prediction_state.get('status') == 'complete' and _test_prediction_state.get('signature') == _test_prediction_signature and _resume_prediction_complete(METHOD, test_stems) and _resume_retention_guard_complete(test_stems))

if _test_prediction_ready:
    predict_seconds = float(_test_prediction_state.get('predict_seconds', 0.0))

    if TEST_PREDICTION_LOG_PATH.is_file():
        _replay_stage_output(TEST_PREDICTION_LOG_PATH)
    else:
        print(f'Prediction completed in {predict_seconds / 60:.2f} minutes')
else:
    remove_path(REPO_DIR / 'predictions')

    for _guard_old_log in WORKING_DIR.glob('retention_guard_*.jsonl'):
        _guard_old_log.unlink()

    for _coordinate_old_log in WORKING_DIR.glob('detector_coordinates_*.jsonl'):
        _coordinate_old_log.unlink()

    for _dependent_path in [BASE_SUBMISSION_STATE_PATH, VALIDATOR_PREDICTION_STATE_PATH, VALIDATOR_BASE_STATE_PATH, PPSWEEP_STATE_PATH, FINAL_SUBMISSION_STATE_PATH]:
        _unlink_resume_file(_dependent_path)

    for _dependent_log in [BASE_SUBMISSION_LOG_PATH, VALIDATOR_PREDICTION_LOG_PATH, VALIDATOR_BASE_LOG_PATH, PPSWEEP_LOG_PATH, FINAL_SUBMISSION_LOG_PATH]:
        _unlink_resume_file(_dependent_log)

    with _capture_stage_output(TEST_PREDICTION_LOG_PATH):
        start_time = time.time()
        available_gpu_count = _torch.cuda.device_count()
        worker_count = min(2, available_gpu_count, len(test_stems))

        if worker_count >= 2 and (not SLICE):
            cuda_tokens = _visible_cuda_tokens(worker_count)
            processes: dict[int, subprocess.Popen] = {}
            commands: dict[int, list[str]] = {}
            print(f'Launching {worker_count} independent video shards on CUDA devices {cuda_tokens}')

            for shard_index in range(worker_count):
                shard_method = f'{METHOD}_gpu{shard_index}'
                shard_cmd = [*predict_cmd, '--method', shard_method, '--slice', f'{shard_index}::{worker_count}']
                shard_env = {**os.environ, 'PYTHONPATH': 'src'}
                shard_env['CUDA_VISIBLE_DEVICES'] = cuda_tokens[shard_index]
                shard_env['BIOHUB_GPU_SHARD'] = f'{shard_index}/{worker_count}'
                print(f'GPU shard {shard_index}: CUDA_VISIBLE_DEVICES = {cuda_tokens[shard_index]} | ' + ' '.join(shard_cmd), flush = True)
                commands[shard_index] = shard_cmd
                processes[shard_index] = _popen_streamed(shard_cmd, cwd = REPO_DIR, env = shard_env)
            _wait_for_prediction_shards(processes, commands)
            _merge_prediction_shards(worker_count)

        else:
            reason = 'SLICE is active' if SLICE else f'only {available_gpu_count} CUDA device(s) available'
            print(f'Using single-process prediction because {reason}.')
            print(' '.join(predict_cmd))
            _run_subprocess_streamed(predict_cmd, cwd = REPO_DIR, env = {**os.environ, 'PYTHONPATH': 'src'})

        predict_seconds = time.time() - start_time
        print(f'Prediction completed in {predict_seconds / 60:.2f} minutes')

    if not _resume_prediction_complete(METHOD, test_stems):
        raise RuntimeError('Test prediction stage finished without a complete prediction cache')

    if not _resume_retention_guard_complete(test_stems):
        raise RuntimeError('Test prediction stage finished without complete frame-retention diagnostics')
    _write_resume_json(TEST_PREDICTION_STATE_PATH, {'resume_schema': RESUME_SCHEMA_VERSION, 'status': 'complete', 'signature': _test_prediction_signature, 'predict_seconds': predict_seconds, 'test_stems': test_stems, 'method': METHOD})

import tracksdata as td
import numpy as np
import blosc2
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

SUBMISSION_COLUMNS = ['dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
CSV_COLUMNS = ['id', *SUBMISSION_COLUMNS]
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)

# Load a geff tracking graph into the internal graph representation
def graph_from_geff(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph

# Measure the physical distance between two graph nodes connected by an edge
def edge_distance_um(source: dict[str, object], target: dict[str, object]) -> float:
    dz = (float(source['z']) - float(target['z'])) * VOXEL_SCALE_UM[0]
    dy = (float(source['y']) - float(target['y'])) * VOXEL_SCALE_UM[1]
    dx = (float(source['x']) - float(target['x'])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)

# Measure the physical distance between two 3d points
def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_SCALE_UM[0]
    dy = (a[1] - b[1]) * VOXEL_SCALE_UM[1]
    dx = (a[2] - b[2]) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)

# Extract a node position as a numeric 3d point
def node_point(node: dict[str, object]) -> tuple[float, float, float]:
    return (float(node['z']), float(node['y']), float(node['x']))

# Build a stable ordering key for graph edges
def edge_sort_key(edge: dict[str, object]) -> tuple[float, float]:
    prob = edge.get('edge_prob')
    prob_value = float(prob) if prob is not None else 0.0
    return (prob_value, -float(edge['distance_um']))

# Generate the next unused node identifier
def _next_node_id(nodes_by_id: dict[int, dict[str, object]]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1

# Load one test frame from the source movie
def read_test_frame(dataset: str, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:
    if t in frame_cache:
        return frame_cache[t]
    zarr_path = TEST_DIR / f'{dataset}.zarr'
    meta = json.loads((zarr_path / '0' / 'zarr.json').read_text())
    shape = tuple((int(v) for v in meta['shape']))
    dtype = np.dtype(meta['data_type'])
    frame_shape = shape[1:]
    chunk_path = zarr_path / '0' / 'c' / str(t) / '0' / '0' / '0'

    try:
        raw = chunk_path.read_bytes()
        arr = np.frombuffer(blosc2.decompress(raw), dtype = dtype)

        if arr.size == int(np.prod(frame_shape)):
            frame = arr.reshape(frame_shape).copy()
            frame_cache[t] = frame
            return frame
    except Exception:
        pass
    import zarr
    frame = np.asarray(zarr.open(zarr_path / '0', mode = 'r')[t])
    frame_cache[t] = frame
    return frame

# Refine a synthetic gap midpoint using local image evidence
def refine_synthetic_midpoint(dataset: str | None, t: int, midpoint: tuple[float, float, float], frame_cache: dict[int, np.ndarray], stats: dict[str, int]) -> tuple[float, float, float]:
    if not GAP_REFINE_SYNTHETIC or dataset is None:
        return midpoint

    try:
        frame = read_test_frame(dataset, t, frame_cache)
        z, y, x = [int(round(v)) for v in midpoint]
        z0 = max(0, z - GAP_REFINE_WIN_Z)
        z1 = min(frame.shape[0], z + GAP_REFINE_WIN_Z + 1)
        y0 = max(0, y - GAP_REFINE_WIN_YX)
        y1 = min(frame.shape[1], y + GAP_REFINE_WIN_YX + 1)
        x0 = max(0, x - GAP_REFINE_WIN_YX)
        x1 = min(frame.shape[2], x + GAP_REFINE_WIN_YX + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)

        if patch.size == 0:
            stats['gap_refine_failed'] += 1
            return midpoint
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())

        if total <= 0:
            stats['gap_refine_failed'] += 1
            return midpoint
        zz = np.arange(z0, z1, dtype = np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype = np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype = np.float64)[None, None, :]
        refined = (float((weights * zz).sum() / total), float((weights * yy).sum() / total), float((weights * xx).sum() / total))

        if point_distance_um(refined, midpoint) > GAP_REFINE_MAX_SHIFT_UM:
            stats['gap_refine_rejected_shift'] += 1
            return midpoint
        stats['gap_refined_synthetic'] += 1
        return refined
    except Exception:
        stats['gap_refine_failed'] += 1
        return midpoint

# Pool a frame in the xy plane for deepcenter inference
def _dc_pool_frame_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy = False)
    z, y, x = volume.shape
    y2 = y // factor * factor
    x2 = x // factor * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy = False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis = (2, 4))

# Normalize frame intensity before deepcenter scoring
def _dc_normalize_dynamic_range(volume: np.ndarray, cfg: object) -> np.ndarray:
    vol = np.asarray(volume, dtype = np.float32)
    lo = float(np.percentile(vol, float(getattr(cfg, 'norm_lo_pct', 50.0))))
    hi = float(np.percentile(vol, float(getattr(cfg, 'norm_hi_pct', 99.5))))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype = np.float32)
    ratio = (vol - lo) / (hi - lo)
    return np.clip(ratio, float(getattr(cfg, 'norm_clip_lo', -0.5)), float(getattr(cfg, 'norm_clip_hi', 6.0))).astype(np.float32)

# Extract candidate deepcenter checkpoint paths from an artifact manifest
def _dc_manifest_weight_paths(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        return []

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        print('Could not read DeepCenter manifest:', manifest_path, type(exc).__name__, exc)
        return []
    root = manifest_path.parent
    sections: list[dict[str, object]] = []

    for section in [manifest.get('model', {}), manifest.get('models', {}).get('full_frame_center', {}) if isinstance(manifest.get('models', {}), dict) else {}, manifest.get('full_frame_center', {})]:
        if isinstance(section, dict):
            sections.append(section)
    candidates: list[Path] = []

    for section in sections:
        for key in ('weight_path', 'path'):
            rel = section.get(key)

            if isinstance(rel, str) and rel:
                candidates.append(root / rel)

        for key in ('last_checkpoint', 'best_checkpoint'):
            item = section.get(key)

            if isinstance(item, dict):
                rel = item.get('path')

                if isinstance(rel, str) and rel:
                    candidates.append(root / rel)

    for name in ('checkpoint_last.pt', 'best.pt', 'last.pt'):
        candidates.append(root / 'weights' / 'full_frame_center' / name)
        candidates.append(root / name)
    candidates.append(root / DEEPCENTER_RELATIVE)
    return candidates

# Build the ordered list of deepcenter checkpoint candidates
def _dc_checkpoint_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get('BIOHUB_DEEPCENTER_CHECKPOINT', DEEPCENTER_CHECKPOINT_DEFAULT).strip()

    if explicit:
        candidates.append(Path(explicit))
    manifest_explicit = os.environ.get('BIOHUB_DEEPCENTER_MANIFEST', DEEPCENTER_MANIFEST_DEFAULT).strip()

    if manifest_explicit:
        candidates.extend(_dc_manifest_weight_paths(Path(manifest_explicit)))
    input_root = Path('/kaggle/input')
    preferred_dirs = [Path('/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1'), Path('/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1')]

    for directory in preferred_dirs:
        candidates.extend(_dc_manifest_weight_paths(directory / 'ARTIFACT_MANIFEST.json'))

        for name in ('checkpoint_last.pt', 'best.pt', 'last.pt'):
            candidates.append(directory / 'weights' / 'full_frame_center' / name)
            candidates.append(directory / name)

    if input_root.exists():
        for name in ('checkpoint_last.pt', 'best.pt', 'last.pt'):
            candidates.extend(sorted(input_root.glob(f'**/full_frame_center/**/{name}')))
    seen: set[Path] = set()
    out: list[Path] = []

    for path in candidates:
        path = path.expanduser()

        try:
            key = path.resolve() if path.exists() else path
        except Exception:
            key = path

        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out

try:
    import torch

except Exception as _dc_torch_error:
    torch = None

if torch is not None:

    # Build the reusable 3d convolution block for the deepcenter network
    class _DCConvBlock3d(torch.nn.Module):

        # Initialize the model layers and configuration
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = min(8, out_channels)
            self.block = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 3, padding = 1, bias = False), torch.nn.GroupNorm(groups, out_channels), torch.nn.SiLU(inplace = True), torch.nn.Conv3d(out_channels, out_channels, 3, padding = 1, bias = False), torch.nn.GroupNorm(groups, out_channels), torch.nn.SiLU(inplace = True))

        # Run the forward pass through the model
        def forward(self, x):
            return self.block(x)

    # Build the 3d deepcenter u-net used as a repair confidence gate
    class _DCDeepCenterUNet3D(torch.nn.Module):

        # Initialize the model layers and configuration
        def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
            super().__init__()
            c = int(base_channels)
            self.enc1 = _DCConvBlock3d(in_channels, c)
            self.down1 = torch.nn.MaxPool3d(2, 2)
            self.enc2 = _DCConvBlock3d(c, c * 2)
            self.down2 = torch.nn.MaxPool3d(2, 2)
            self.enc3 = _DCConvBlock3d(c * 2, c * 4)
            self.down3 = torch.nn.MaxPool3d(2, 2)
            self.bottleneck = _DCConvBlock3d(c * 4, c * 8)
            self.up3 = torch.nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
            self.dec3 = _DCConvBlock3d(c * 8, c * 4)
            self.up2 = torch.nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
            self.dec2 = _DCConvBlock3d(c * 4, c * 2)
            self.up1 = torch.nn.ConvTranspose3d(c * 2, c, 2, 2)
            self.dec1 = _DCConvBlock3d(c * 2, c)
            self.head = torch.nn.Conv3d(c, 1, 1)

        # Run the forward pass through the model
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            b = self.bottleneck(self.down3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim = 1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim = 1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim = 1))
            return self.head(d1)

else:
    _DCConvBlock3d = None
    _DCDeepCenterUNet3D = None

# Load and validate the deepcenter repair-gating model
def load_deepcenter_veto_detector() -> dict[str, object] | None:
    if not USE_DEEPCENTER_VETO:
        print('DeepCenter add-only repair gate disabled by configuration.')
        return None

    if torch is None:
        if REQUIRE_DEEPCENTER_VETO:
            raise ImportError('torch is required for DeepCenter add-only repair gate')
        print('DeepCenter add-only repair gate skipped because torch is unavailable.')
        return None
    from types import SimpleNamespace
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_errors: list[str] = []

    for checkpoint_path in _dc_checkpoint_candidates():
        if not checkpoint_path.exists():
            continue

        try:
            print('Trying DeepCenter add-only gate checkpoint:', checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = False)

            if not isinstance(checkpoint, dict) or 'model_state' not in checkpoint:
                raise ValueError('checkpoint has no model_state')
            checkpoint_epoch = int(checkpoint.get('epoch', -1))

            if DEEPCENTER_EXPECTED_EPOCH > 0 and checkpoint_epoch != DEEPCENTER_EXPECTED_EPOCH:
                raise ValueError(f'expected DeepCenter epoch {DEEPCENTER_EXPECTED_EPOCH}, got {checkpoint_epoch}')
            cfg = SimpleNamespace(**checkpoint.get('config', {}))
            model = _DCDeepCenterUNet3D(base_channels = int(getattr(cfg, 'base_channels', 24)))
            model.load_state_dict(checkpoint['model_state'])
            model.to(device)
            model.eval()
            print('Loaded DeepCenter add-only gate checkpoint:', checkpoint_path)
            print('DeepCenter checkpoint epoch:', checkpoint.get('epoch'), 'best_score:', checkpoint.get('best_score'))
            return {'model': model, 'cfg': cfg, 'device': device, 'path': checkpoint_path, 'torch': torch}
        except Exception as exc:
            load_errors.append(f'{checkpoint_path}: {type(exc).__name__}: {exc}')
            print('Skipping incompatible DeepCenter checkpoint:', checkpoint_path, '|', type(exc).__name__, exc)
    message = 'No usable DeepCenter checkpoint found for add-only repair gate.'

    if REQUIRE_DEEPCENTER_VETO:
        checked = '\n'.join((str(p) for p in _dc_checkpoint_candidates()[:80]))
        errors = '\n'.join(load_errors[-20:])
        raise FileNotFoundError(message + '\nChecked:\n' + checked + ('\nLoad errors:\n' + errors if errors else ''))
    print(message)
    return None

# Trim cached deepcenter heatmaps to the configured frame limit
def _dc_cache_trim(cache: dict[tuple[str, int], np.ndarray]) -> None:
    limit = max(1, int(DEEPCENTER_SCORE_CACHE_MAX_FRAMES))

    while len(cache) > limit:
        cache.pop(next(iter(cache)))

# Generate or reuse the deepcenter heatmap for a requested frame
def deepcenter_heatmap_for_frame(dataset: str, t: int, detector_bundle: dict[str, object] | None, frame_cache: dict[int, np.ndarray], heatmap_cache: dict[tuple[str, int], np.ndarray]) -> np.ndarray | None:
    if detector_bundle is None:
        return None
    key = (dataset, int(t))
    cached = heatmap_cache.get(key)

    if cached is not None:
        return cached
    model = detector_bundle['model']
    cfg = detector_bundle['cfg']
    device = detector_bundle['device']
    torch_mod = detector_bundle['torch']
    pool_factor = int(getattr(cfg, 'pool_factor', 4))
    volume = read_test_frame(dataset, int(t), frame_cache)
    pooled = _dc_pool_frame_xy(volume, pool_factor)
    image = _dc_normalize_dynamic_range(pooled, cfg)

    with torch_mod.no_grad():
        tensor = torch_mod.from_numpy(image[None, None, ...]).to(device = device, dtype = torch_mod.float32)
        logits = model(tensor)

        if os.environ.get('BIOHUB_DEEPCENTER_TTA', '0') != '0':
            acc = logits.clone()
            nv = 1

            for dims in [(-1,), (-2,), (-2, -1)]:
                acc = acc + model(tensor.flip(dims)).flip(dims)
                nv += 1

            if tensor.shape[-1] == tensor.shape[-2]:
                for k in (1, 3):
                    acc = acc + torch_mod.rot90(model(torch_mod.rot90(tensor, k, dims = (-2, -1))), -k, dims = (-2, -1))
                    nv += 1
                acc = acc + model(tensor.transpose(-1, -2)).transpose(-1, -2)
                nv += 1
                at = torch_mod.rot90(tensor, 1, dims = (-2, -1)).transpose(-1, -2)
                acc = acc + torch_mod.rot90(model(at).transpose(-1, -2), -1, dims = (-2, -1))
                nv += 1
            delta = float((acc / nv - logits).abs().mean())

            if delta == 0.0:
                raise RuntimeError('DeepCenter TTA produced no logit change')

            if not getattr(deepcenter_heatmap_for_frame, '_tta_announced', False):
                print('DEEPCENTER_TTA_ACTIVE views =', nv, 'mean_abs_logit_delta =', round(delta, 6), flush = True)
                deepcenter_heatmap_for_frame._tta_announced = True
            logits = acc / nv
        heatmap = torch_mod.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32, copy = False)
    heatmap_cache[key] = heatmap
    _dc_cache_trim(heatmap_cache)
    return heatmap

# Score a candidate 3d point using the deepcenter heatmap
def deepcenter_score_point(dataset: str | None, t: int, point: tuple[float, float, float], detector_bundle: dict[str, object] | None, frame_cache: dict[int, np.ndarray], heatmap_cache: dict[tuple[str, int], np.ndarray]) -> float | None:
    if not USE_DEEPCENTER_VETO or detector_bundle is None or dataset is None:
        return None
    heatmap = deepcenter_heatmap_for_frame(dataset, int(t), detector_bundle, frame_cache, heatmap_cache)

    if heatmap is None or heatmap.size == 0:
        return None
    cfg = detector_bundle['cfg']
    pool_factor = int(getattr(cfg, 'pool_factor', 4))
    z = int(round(float(point[0])))
    y = int(round(float(point[1]) / max(pool_factor, 1)))
    x = int(round(float(point[2]) / max(pool_factor, 1)))
    z0, z1 = (max(0, z - DEEPCENTER_SCORE_WIN_Z), min(heatmap.shape[0], z + DEEPCENTER_SCORE_WIN_Z + 1))
    y0, y1 = (max(0, y - DEEPCENTER_SCORE_WIN_YX), min(heatmap.shape[1], y + DEEPCENTER_SCORE_WIN_YX + 1))
    x0, x1 = (max(0, x - DEEPCENTER_SCORE_WIN_YX), min(heatmap.shape[2], x + DEEPCENTER_SCORE_WIN_YX + 1))
    patch = heatmap[z0:z1, y0:y1, x0:x1]

    if patch.size == 0:
        return None
    score = float(np.max(patch))
    return score if np.isfinite(score) else None

# Decide whether deepcenter evidence is strong enough to accept a repair
def deepcenter_accept_repair_point(dataset: str | None, t: int, point: tuple[float, float, float], detector_bundle: dict[str, object] | None, frame_cache: dict[int, np.ndarray], heatmap_cache: dict[tuple[str, int], np.ndarray], stats: dict[str, int], prefix: str, threshold: float) -> bool:
    if not USE_DEEPCENTER_VETO:
        return True

    if detector_bundle is None or dataset is None:
        stats[f'deepcenter_{prefix}_missing'] += 1
        return True
    stats[f'deepcenter_{prefix}_checked'] += 1
    score = deepcenter_score_point(dataset, int(t), point, detector_bundle, frame_cache, heatmap_cache)

    if score is None:
        stats[f'deepcenter_{prefix}_missing'] += 1
        return True

    if score < float(threshold):
        stats[f'deepcenter_{prefix}_rejected'] += 1
        return False
    stats[f'deepcenter_{prefix}_accepted'] += 1
    return True

# Convert a graph node position from voxel coordinates to micrometers
def _position_um(node: dict[str, object]) -> np.ndarray:
    return np.array([float(node['z']) * VOXEL_SCALE_UM[0], float(node['y']) * VOXEL_SCALE_UM[1], float(node['x']) * VOXEL_SCALE_UM[2]], dtype = np.float64)

# Recover plausible missing edges using motion consistency and learned edge evidence
def motion_relink_edges(nodes_by_id: dict[int, dict[str, object]], stats: dict[str, int], learned_edge_probs: dict[tuple[int, int], float] | None = None) -> list[dict[str, object]]:
    if not OUTPUT_MOTION_RELINK or not nodes_by_id:
        return []
    learned_edge_probs = learned_edge_probs or {}

    # Read the learned association probability for a candidate edge
    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not np.isfinite(value):
            return 0.0

        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))
    ids_by_t: dict[int, list[int]] = {}

    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node['t']), []).append(node_id)

    for ids in ids_by_t.values():
        ids.sort()
    frame_sizes = [len(ids) for ids in ids_by_t.values()]

    if frame_sizes and max(frame_sizes) > MOTION_RELINK_MAX_FRAME_NODES:
        stats['motion_relink_skipped_large_frame'] = 1
        return []
    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    # Assign unmatched source and target nodes under the current relinking constraints
    def assign_pass(source_ids: list[int], target_ids: list[int], gate_um: float) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype = np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)

        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)

            if prev_pos is None:
                predicted = source_pos
            else:
                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)

            for j, target_id in enumerate(target_ids):
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))

                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target_pos - predicted))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int, float, float, float]] = []

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append((source_ids[int(r)], target_ids[int(c)], float(raw_dist[r, c]), float(motion_dist[r, c]), float(prob_matrix[r, c])))
        return matches
    times = sorted(ids_by_t)

    for t in times:
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])

        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[tuple[int, int, float, float, str, float]] = []

        for pass_name, gate_um in (('tight', MOTION_RELINK_TIGHT_UM), ('relaxed', MOTION_RELINK_RELAXED_UM)):
            pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
            pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
            matches = assign_pass(pass_sources, pass_targets, gate_um)

            for source_id, target_id, raw, motion, prob in matches:
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((source_id, target_id, raw, motion, pass_name, prob))

                if pass_name == 'tight':
                    stats['motion_relink_tight_edges'] += 1
                else:
                    stats['motion_relink_relaxed_edges'] += 1

        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append({'source_id': source_id, 'target_id': target_id, 'edge_prob': prob, 'distance_um': raw, 'motion_distance_um': motion, 'motion_relinked': 1, 'motion_pass': pass_name})
            predecessor_position_um[target_id] = position_um[source_id]
        stats['motion_relink_frames'] += 1
    stats['motion_relink_edges'] = len(selected_edges)
    return selected_edges

# Close eligible one-frame tracking gaps while respecting geometric safeguards
def close_single_frame_gaps(nodes_by_id: dict[int, dict[str, object]], edges: list[dict[str, object]], stats: dict[str, int], dataset: str | None = None, deepcenter_bundle: dict[str, object] | None = None, frame_cache: dict[int, np.ndarray] | None = None, deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP_CLOSE or GAP_CLOSE_MAX_GAP < 1 or (not edges):
        return (nodes_by_id, edges)
    outgoing = {int(edge['source_id']) for edge in edges}
    incoming = {int(edge['target_id']) for edge in edges}
    incident = outgoing | incoming
    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    all_ids_by_t: dict[int, list[int]] = {}

    for node_id, node in nodes_by_id.items():
        t = int(node['t'])
        all_ids_by_t.setdefault(t, []).append(node_id)

        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)

        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)
    max_synthetic = min(GAP_CLOSE_MAX_ADDED_ABS, max(1, int(round(len(nodes_by_id) * GAP_CLOSE_MAX_ADDED_FRAC))) if GAP_CLOSE_MAX_ADDED_FRAC > 0 else 0)
    next_id = _next_node_id(nodes_by_id)
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict[str, object]] = []
    density_cache: dict[int, dict[int, float]] = {}

    # Estimate local inter-cell spacing around a frame position
    def frame_local_spacing(t: int) -> dict[int, float]:
        cached = density_cache.get(t)

        if cached is not None:
            return cached
        frame_ids = all_ids_by_t.get(t, [])

        if len(frame_ids) <= 1:
            result = {node_id: GAP_DENSITY_REFERENCE_UM for node_id in frame_ids}
            density_cache[t] = result
            return result
        positions = np.stack([_position_um(nodes_by_id[node_id]) for node_id in frame_ids])
        tree = cKDTree(positions)
        query_k = min(len(frame_ids), max(2, GAP_DENSITY_NEIGHBORS + 1))
        distances, _ = tree.query(positions, k = query_k)

        if distances.ndim == 1:
            distances = distances[:, None]
        result: dict[int, float] = {}

        for idx, node_id in enumerate(frame_ids):
            neighbour_distances = distances[idx, 1:]
            neighbour_distances = neighbour_distances[np.isfinite(neighbour_distances)]
            spacing = float(np.median(neighbour_distances)) if neighbour_distances.size else GAP_DENSITY_REFERENCE_UM
            result[node_id] = spacing
        density_cache[t] = result
        stats['gap_density_nodes_scored'] += len(result)
        return result
    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)
    stats['gap_close_effective_max_gap'] = effective_gap_max

    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]

            if not end_ids or not start_ids:
                continue
            end_points = [node_point(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_point(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = GAP_CLOSE_UM * (gap + 1)
            d = np.zeros((len(end_ids), len(start_ids)), dtype = np.float64)
            adaptive_threshold = np.full_like(d, threshold_um)
            source_spacing = frame_local_spacing(t)
            target_spacing = frame_local_spacing(t + gap + 1)

            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_distance_um(ep, sp)

                    if GAP_DENSITY_ADAPTIVE:
                        local_spacing = 0.5 * (source_spacing.get(end_ids[i], GAP_DENSITY_REFERENCE_UM) + target_spacing.get(start_ids[j], GAP_DENSITY_REFERENCE_UM))
                        step_delta = float(np.clip(GAP_DENSITY_GAIN * (local_spacing - GAP_DENSITY_REFERENCE_UM), -GAP_DENSITY_MAX_STEP_DELTA_UM, GAP_DENSITY_MAX_STEP_DELTA_UM))
                        adaptive_threshold[i, j] = threshold_um + step_delta * (gap + 1)
                        stats['gap_density_step_delta_milli_sum'] += int(round(1000.0 * step_delta))
            base_allowed = d <= threshold_um
            adaptive_allowed = d <= adaptive_threshold
            stats['gap_density_candidates_expanded'] += int((adaptive_allowed & ~base_allowed).sum())
            stats['gap_density_candidates_restricted'] += int((base_allowed & ~adaptive_allowed).sum())
            stats['gap_candidates'] += int(adaptive_allowed.sum())

            if not np.isfinite(d).any():
                continue
            max_threshold = float(np.max(adaptive_threshold))
            big = max_threshold * 1000.0 + 1.0
            cost = np.where(adaptive_allowed, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if not adaptive_allowed[r, c]:
                    continue

                if not base_allowed[r, c]:
                    stats['gap_density_selected_outside_base'] += 1
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]

                if source_id in outgoing or target_id in used_starts:
                    continue
                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source['t']) + gap
                mid_point = ((float(source['z']) + float(target['z'])) / 2.0, (float(source['y']) + float(target['y'])) / 2.0, (float(source['x']) + float(target['x'])) / 2.0)
                middle_id: int | None = None
                middle_reused = False

                if GAP_CLOSE_REUSE_EXISTING:
                    candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]

                    if candidates:
                        distances = [point_distance_um(node_point(nodes_by_id[nid]), mid_point) for nid in candidates]
                        best_idx = int(np.argmin(distances))

                        if distances[best_idx] <= GAP_CLOSE_REUSE_UM:
                            middle_id = candidates[best_idx]
                            middle_reused = True

                if middle_id is None:
                    if synthetic_added >= max_synthetic:
                        stats['gap_skipped_node_cap'] += 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    refined_point = refine_synthetic_midpoint(dataset, mid_t, mid_point, frame_cache, stats)
                    nodes_by_id[middle_id] = {'node_id': middle_id, 't': mid_t, 'z': refined_point[0], 'y': refined_point[1], 'x': refined_point[2], 'gap_synthetic': 1}
                    synthetic_added += 1
                    stats['gap_inserted_synthetic'] += 1
                middle = nodes_by_id[middle_id]
                gap_span_um = float(d[r, c])
                marginal_gap = gap_span_um >= DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM
                synthetic_middle = int(middle.get('gap_synthetic', 0)) == 1
                requires_center_confirmation = DEEPCENTER_GAP_VETO and marginal_gap and synthetic_middle

                if DEEPCENTER_GAP_VETO and (not marginal_gap):
                    stats['deepcenter_gap_bypassed_strong_motion'] += 1
                elif DEEPCENTER_GAP_VETO and (not synthetic_middle):
                    stats['deepcenter_gap_bypassed_observed_node'] += 1

                if requires_center_confirmation and (not deepcenter_accept_repair_point(dataset, mid_t, node_point(middle), deepcenter_bundle, frame_cache, deepcenter_cache, stats, 'gap', DEEPCENTER_GAP_THRESHOLD)):
                    if int(middle.get('gap_synthetic', 0)) == 1:
                        nodes_by_id.pop(middle_id, None)
                        synthetic_added = max(0, synthetic_added - 1)
                        stats['gap_inserted_synthetic'] = max(0, stats['gap_inserted_synthetic'] - 1)
                    continue

                if middle_reused:
                    used_isolated.add(middle_id)
                    stats['gap_reused_existing'] += 1
                e1 = {'source_id': source_id, 'target_id': middle_id, 'edge_prob': None, 'distance_um': edge_distance_um(source, middle), 'gap_closed': 1}
                e2 = {'source_id': middle_id, 'target_id': target_id, 'edge_prob': None, 'distance_um': edge_distance_um(middle, target), 'gap_closed': 1}
                new_edges.extend([e1, e2])
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats['gap_pairs_selected'] += 1
                stats['gap_added_edges'] += 2

    if new_edges:
        edges = [*edges, *new_edges]
    stats['gap_added_nodes'] = stats['gap_inserted_synthetic']
    return (nodes_by_id, edges)

# Build a node-to-successor map for nodes with exactly one outgoing edge
def _single_successor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}

    for edge in edges:
        by_source.setdefault(int(edge['source_id']), []).append(int(edge['target_id']))
    return {source: targets[0] for source, targets in by_source.items() if len(targets) == 1}

# Build a node-to-predecessor map for nodes with exactly one incoming edge
def _single_predecessor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}

    for edge in edges:
        by_target.setdefault(int(edge['target_id']), []).append(int(edge['source_id']))
    return {target: sources[0] for target, sources in by_target.items() if len(sources) == 1}

# Recover strict two-frame gaps only when temporal context supports the repair
def recover_strict_gap2(nodes_by_id: dict[int, dict[str, object]], edges: list[dict[str, object]], stats: dict[str, int], dataset: str | None = None) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP2_RECOVERY or not edges or (not nodes_by_id):
        return (nodes_by_id, edges)
    outgoing = {int(edge['source_id']) for edge in edges}
    incoming = {int(edge['target_id']) for edge in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)
    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}

    for node_id, node in nodes_by_id.items():
        t = int(node['t'])

        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)

        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
    cap = min(GAP2_MAX_LINKS_ABS, max(1, int(round(len(edges) * GAP2_MAX_LINKS_FRAC))))
    proposals: list[tuple[float, int, int, int, float]] = []

    # Convert a node identifier into its physical 3d position
    def pos_um(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        return np.array([float(node['z']), float(node['y']), float(node['x'])], dtype = np.float64) * np.array(VOXEL_SCALE_UM)

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])

        if not end_ids or not start_ids:
            continue

        for end_id in end_ids:
            end_pos = pos_um(end_id)

            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))

                if dist > GAP2_MAX_TOTAL_UM or dist / 3.0 > GAP2_MAX_STEP_UM:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0

                if GAP2_REQUIRE_CONTEXT:
                    ok_context = False
                    prev_id = predecessor.get(end_id)

                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))

                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-09))

                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    next_id = successor.get(start_id)

                    if next_id is not None:
                        next_step = pos_um(next_id) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))

                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-09))

                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)

                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))
    proposals.sort(key = lambda item: item[0])
    stats['gap2_candidates'] = len(proposals)

    if not proposals:
        return (nodes_by_id, edges)
    selected: list[tuple[float, int, int, int, float]] = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}

    for proposal in proposals:
        if len(selected) >= cap:
            stats['gap2_skipped_cap'] += 1
            break
        _, end_id, start_id, t, _ = proposal

        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * GAP2_FRAME_FRAC_CAP)))

        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1

    if not selected:
        return (nodes_by_id, edges)
    next_node_id = _next_node_id(nodes_by_id)
    frame_cache: dict[int, np.ndarray] = {}
    new_edges: list[dict[str, object]] = []

    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted_ids: list[int] = []

        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source['t']) + k
            midpoint = (float(source['z']) + (float(target['z']) - float(source['z'])) * frac, float(source['y']) + (float(target['y']) - float(source['y'])) * frac, float(source['x']) + (float(target['x']) - float(source['x'])) * frac)
            refined_point = refine_synthetic_midpoint(dataset, mid_t, midpoint, frame_cache, stats)
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {'node_id': node_id, 't': mid_t, 'z': refined_point[0], 'y': refined_point[1], 'x': refined_point[2]}
            inserted_ids.append(node_id)
            current = nodes_by_id[node_id]
            new_edges.append({'source_id': previous_id, 'target_id': node_id, 'edge_prob': None, 'distance_um': edge_distance_um(nodes_by_id[previous_id], current), 'gap2_recovered': 1})
            previous_id = node_id
        new_edges.append({'source_id': previous_id, 'target_id': start_id, 'edge_prob': None, 'distance_um': edge_distance_um(nodes_by_id[previous_id], target), 'gap2_recovered': 1})
        stats['gap2_pairs_selected'] += 1
        stats['gap2_added_nodes'] += len(inserted_ids)
        stats['gap2_added_edges'] += 3
    return (nodes_by_id, [*edges, *new_edges])

# Add conservative division edges after ordinary temporal linking is complete
def add_safe_divisions_postlink(nodes_by_id: dict[int, dict[str, object]], edges: list[dict[str, object]], stats: dict[str, int], dataset: str | None = None, deepcenter_bundle: dict[str, object] | None = None, frame_cache: dict[int, np.ndarray] | None = None, deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None) -> list[dict[str, object]]:
    if not OUTPUT_SAFE_DIVISIONS or not edges or (not nodes_by_id):
        return edges
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()

    for edge in edges:
        out_by_source.setdefault(int(edge['source_id']), []).append(edge)
        incoming.add(int(edge['target_id']))
    ids_by_t: dict[int, list[int]] = {}

    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node['t']), []).append(node_id)
    existing_edges = {(int(edge['source_id']), int(edge['target_id'])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()
    used_sources: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])

        if not child_frame_ids:
            continue
        source_ids = [node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1]
        candidate_ids = [node_id for node_id in child_frame_ids if node_id not in incoming and node_id not in used_targets]

        if not source_ids or not candidate_ids:
            continue
        candidate_tree = None

        if SAFE_DIV_REQUIRE_MUTUAL_NN:
            candidate_positions = np.stack([_position_um(nodes_by_id[cid]) for cid in candidate_ids])
            candidate_tree = cKDTree(candidate_positions)
        frame_cap = max(1, int(round(len(source_ids) * SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list[tuple[float, int, int, float, float]] = []

        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge['target_id'])
            existing_child = nodes_by_id.get(existing_child_id)

            if existing_child is None or int(existing_child['t']) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)

            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue
            mutual_nn_id = None

            if candidate_tree is not None:
                _, nn_idx = candidate_tree.query(_position_um(existing_child))
                mutual_nn_id = candidate_ids[int(nn_idx)]

            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)

                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)

                if sister_dist > SAFE_DIV_SISTER_MAX_UM:
                    continue

                if SAFE_DIV_REQUIRE_MUTUAL_NN and candidate_id != mutual_nn_id:
                    stats['safe_division_mutual_nn_rejected'] += 1
                    continue

                if SAFE_DIV_REQUIRE_DIVERGENCE:
                    c1_succ = out_by_source.get(existing_child_id, [])
                    q_succ = out_by_source.get(candidate_id, [])

                    if len(c1_succ) != 1 or len(q_succ) != 1:
                        stats['safe_division_divergence_rejected'] += 1
                        continue
                    c1_grandchild = nodes_by_id.get(int(c1_succ[0]['target_id']))
                    q_grandchild = nodes_by_id.get(int(q_succ[0]['target_id']))

                    if c1_grandchild is None or q_grandchild is None or int(c1_grandchild['t']) != t + 2 or (int(q_grandchild['t']) != t + 2):
                        stats['safe_division_divergence_rejected'] += 1
                        continue
                    grandchild_dist = edge_distance_um(c1_grandchild, q_grandchild)

                    if grandchild_dist - sister_dist < SAFE_DIV_DIVERGE_UM:
                        stats['safe_division_divergence_rejected'] += 1
                        continue
                stats['safe_division_geometric_candidates'] += 1

                if DEEPCENTER_SAFE_DIV_VETO and (not deepcenter_accept_repair_point(dataset, int(candidate['t']), node_point(candidate), deepcenter_bundle, frame_cache, deepcenter_cache, stats, 'safe_div', DEEPCENTER_SAFE_DIV_THRESHOLD)):
                    continue

                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:
                    symmetry_denominator = max((child_dist + parent_dist) / 2.0, 1e-6)

                    if abs(child_dist - parent_dist) / symmetry_denominator > SAFE_DIV_SISTER_SYMMETRY_TAU:
                        stats['safe_division_symmetry_rejected'] += 1
                        continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))
        stats['safe_division_candidates'] += len(proposals)

        if not proposals:
            continue
        proposals.sort(key = lambda item: item[0])
        added_this_frame = 0

        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                stats['safe_division_skipped_cap'] += 1
                break

            if added_this_frame >= frame_cap:
                break

            if candidate_id in used_targets or candidate_id in incoming:
                continue

            if source_id in used_sources:
                continue
            candidate = nodes_by_id[candidate_id]
            added.append({'source_id': source_id, 'target_id': candidate_id, 'edge_prob': None, 'distance_um': parent_dist, 'safe_division': 1})
            used_targets.add(candidate_id)
            used_sources.add(source_id)
            added_this_frame += 1

    if added:
        stats['safe_divisions_added'] = len(added)
        return [*edges, *added]
    return edges

# Remove weak short-track components while preserving protected lineage structures
def filter_short_track_components(nodes_by_id: dict[int, dict[str, object]], edges: list[dict[str, object]], stats: dict[str, int]) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_FILTER_SHORT_TRACKS or OUTPUT_MIN_TRACK_LEN <= 1 or (not edges):
        return (nodes_by_id, edges)
    parent = {node_id: node_id for node_id in nodes_by_id}

    # Find the canonical disjoint-set representative for a node
    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    # Merge two disjoint-set components during connectivity construction
    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[ra] = rb
    out_count: dict[int, int] = {}

    for edge in edges:
        source_id = int(edge['source_id'])
        target_id = int(edge['target_id'])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1
    components: dict[int, list[int]] = {}

    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)
    component_edges: dict[int, list[dict[str, object]]] = {root: [] for root in components}

    for edge in edges:
        source_id = int(edge['source_id'])
        target_id = int(edge['target_id'])

        if source_id in parent and target_id in parent:
            component_edges.setdefault(find(source_id), []).append(edge)
    keep: set[int] = set()

    for root, members in components.items():
        has_division = any((out_count.get(node_id, 0) >= 2 for node_id in members))

        if len(members) >= OUTPUT_MIN_TRACK_LEN or (OUTPUT_KEEP_DIVISION_COMPONENTS and has_division):
            keep.update(members)

    if not keep:
        stats['short_track_filter_skipped_all'] += 1
        return (nodes_by_id, edges)
    removed_before_rescue = len(nodes_by_id) - len(keep)

    if removed_before_rescue <= 0:
        return (nodes_by_id, edges)

    if ADAPTIVE_SHORT_TRACK_RESCUE:
        removed_frac = removed_before_rescue / max(len(nodes_by_id), 1)

        if removed_frac >= SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC:
            budget = min(SHORT_TRACK_RESCUE_MAX_NODES_ABS, max(0, int(round(len(nodes_by_id) * SHORT_TRACK_RESCUE_MAX_NODES_FRAC))))
            stats['short_track_rescue_triggered'] = 1
            stats['short_track_rescue_budget'] = budget
            proposals: list[tuple[float, int, float, int, list[int]]] = []

            for root, members in components.items():
                if set(members) & keep:
                    continue

                if len(members) < SHORT_TRACK_RESCUE_MIN_LEN or len(members) >= OUTPUT_MIN_TRACK_LEN:
                    continue
                c_edges = component_edges.get(root, [])

                if not c_edges:
                    continue
                probs: list[float] = []
                dists: list[float] = []

                for edge in c_edges:
                    try:
                        prob = float(edge.get('edge_prob', 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0

                    if np.isfinite(prob):
                        probs.append(prob)

                    try:
                        dist = float(edge.get('distance_um', np.nan))
                    except (TypeError, ValueError):
                        dist = np.nan

                    if np.isfinite(dist):
                        dists.append(dist)
                mean_prob = float(np.mean(probs)) if probs else 0.0
                mean_dist = float(np.mean(dists)) if dists else float('inf')

                if mean_prob < SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB:
                    continue

                if mean_dist > SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM:
                    continue
                score = mean_prob - 0.02 * mean_dist + 0.004 * len(members)
                proposals.append((score, len(members), mean_prob, root, members))
            proposals.sort(reverse = True)
            rescued_nodes = 0
            rescued_components = 0

            for _, size, _, _, members in proposals:
                if budget <= 0 or rescued_nodes + size > budget:
                    continue
                keep.update(members)
                rescued_nodes += size
                rescued_components += 1
            stats['short_track_rescue_components'] = rescued_components
            stats['short_track_rescue_nodes'] = rescued_nodes
    removed_nodes = len(nodes_by_id) - len(keep)

    if removed_nodes <= 0:
        return (nodes_by_id, edges)
    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in keep}
    kept_edges = [edge for edge in edges if int(edge['source_id']) in kept_nodes and int(edge['target_id']) in kept_nodes]
    stats['short_track_components_removed'] = sum((1 for members in components.values() if not set(members) & keep))
    stats['short_track_nodes_removed'] = removed_nodes
    stats['short_track_edges_removed'] = len(edges) - len(kept_edges)
    return (kept_nodes, kept_edges)

# Smooth eligible track coordinates with a local line fit before export
def linefit_smooth_output_graph(nodes_by_id: dict[int, dict[str, object]], edges: list[dict[str, object]], stats: dict[str, int]) -> dict[int, dict[str, object]]:
    """Smooth linear track interiors without changing graph topology."""

    if not OUTPUT_LINEFIT_SMOOTH or OUTPUT_LINEFIT_WEIGHT <= 0 or OUTPUT_LINEFIT_WINDOW <= 0 or (not edges):
        return nodes_by_id
    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}

    for edge in edges:
        source_id = int(edge['source_id'])
        target_id = int(edge['target_id'])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)

        if source is None or target is None:
            continue

        if int(target['t']) != int(source['t']) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)
    original_pos = {node_id: np.array([float(node['z']), float(node['y']), float(node['x'])], dtype = np.float64) for node_id, node in nodes_by_id.items()}
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]
        current = node_id

        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            prev_ids = predecessor.get(current, [])

            if len(prev_ids) != 1:
                break
            current = prev_ids[0]

            if current not in original_pos:
                break
            neighbourhood.append((-step, current))
        current = node_id

        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            next_ids = successor.get(current, [])

            if len(next_ids) != 1:
                break
            current = next_ids[0]

            if current not in original_pos:
                break
            neighbourhood.append((step, current))

        if len(neighbourhood) < 3:
            stats['linefit_skipped_nodes'] += 1
            continue
        dts = np.array([delta for delta, _ in neighbourhood], dtype = np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)], dtype = np.float64)

        if not np.isfinite(fitted).all():
            stats['linefit_skipped_nodes'] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]['z'] = float(pos[0])
        nodes_by_id[node_id]['y'] = float(pos[1])
        nodes_by_id[node_id]['x'] = float(pos[2])
    stats['linefit_smoothed_nodes'] = len(updated_pos)
    return nodes_by_id

# Apply final graph validity filters and conservative repair rules
def filter_output_graph(nodes_by_id: dict[int, dict[str, object]], raw_edges: list[dict[str, object]], dataset: str | None = None, deepcenter_bundle: dict[str, object] | None = None) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]:
    stats = {'raw_edges': len(raw_edges), 'dropped_nonconsecutive_edges': 0, 'dropped_long_edges': 0, 'dropped_multi_parent_edges': 0, 'dropped_multi_child_edges': 0, 'dropped_division_edges': 0, 'gap_candidates': 0, 'gap_pairs_selected': 0, 'gap_reused_existing': 0, 'gap_inserted_synthetic': 0, 'gap_added_nodes': 0, 'gap_added_edges': 0, 'gap_skipped_node_cap': 0, 'gap_density_nodes_scored': 0, 'gap_density_candidates_expanded': 0, 'gap_density_candidates_restricted': 0, 'gap_density_selected_outside_base': 0, 'gap_density_step_delta_milli_sum': 0, 'gap_refined_synthetic': 0, 'gap_refine_failed': 0, 'gap_refine_rejected_shift': 0, 'pruned_isolated_nodes': 0, 'motion_relink_edges': 0, 'motion_relink_tight_edges': 0, 'motion_relink_relaxed_edges': 0, 'motion_relink_frames': 0, 'motion_relink_replaced_raw_edges': 0, 'motion_relink_fallback_raw': 0, 'motion_relink_skipped_large_frame': 0, 'gap2_candidates': 0, 'gap2_pairs_selected': 0, 'gap2_added_nodes': 0, 'gap2_added_edges': 0, 'gap2_skipped_cap': 0, 'safe_division_candidates': 0, 'safe_division_geometric_candidates': 0, 'safe_divisions_added': 0, 'safe_division_skipped_cap': 0, 'safe_division_mutual_nn_rejected': 0, 'safe_division_divergence_rejected': 0, 'safe_division_symmetry_rejected': 0, 'deepcenter_gap_checked': 0, 'deepcenter_gap_bypassed_strong_motion': 0, 'deepcenter_gap_bypassed_observed_node': 0, 'deepcenter_gap_accepted': 0, 'deepcenter_gap_rejected': 0, 'deepcenter_gap_missing': 0, 'deepcenter_safe_div_checked': 0, 'deepcenter_safe_div_accepted': 0, 'deepcenter_safe_div_rejected': 0, 'deepcenter_safe_div_missing': 0, 'short_track_components_removed': 0, 'short_track_nodes_removed': 0, 'short_track_edges_removed': 0, 'short_track_filter_skipped_all': 0, 'short_track_rescue_triggered': 0, 'short_track_rescue_components': 0, 'short_track_rescue_nodes': 0, 'short_track_rescue_budget': 0, 'linefit_smoothed_nodes': 0, 'linefit_skipped_nodes': 0}
    edges: list[dict[str, object]] = []

    for edge in raw_edges:
        source = nodes_by_id.get(int(edge['source_id']))
        target = nodes_by_id.get(int(edge['target_id']))

        if source is None or target is None:
            continue

        if OUTPUT_ENFORCE_NEXT_FRAME and int(target['t']) != int(source['t']) + 1:
            stats['dropped_nonconsecutive_edges'] += 1
            continue
        distance_um = edge_distance_um(source, target)
        edge['distance_um'] = distance_um

        if OUTPUT_EDGE_MAX_UM > 0 and distance_um > OUTPUT_EDGE_MAX_UM:
            stats['dropped_long_edges'] += 1
            continue
        edges.append(edge)

    if OUTPUT_MOTION_RELINK:
        learned_edge_probs: dict[tuple[int, int], float] = {}

        for edge in edges:
            prob = edge.get('edge_prob')

            if prob is None:
                continue

            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue

            if np.isfinite(prob):
                key = (int(edge['source_id']), int(edge['target_id']))
                learned_edge_probs[key] = max(learned_edge_probs.get(key, float('-inf')), prob)
        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)

        if motion_edges:
            stats['motion_relink_replaced_raw_edges'] = len(edges)
            edges = motion_edges
        else:
            stats['motion_relink_fallback_raw'] = 1

    if OUTPUT_SINGLE_PARENT_REPAIR and edges:
        best_by_target: dict[int, dict[str, object]] = {}

        for edge in edges:
            target_id = int(edge['target_id'])
            prev = best_by_target.get(target_id)

            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(edge) for edge in best_by_target.values()}
        stats['dropped_multi_parent_edges'] = sum((1 for edge in edges if id(edge) not in kept_ids))
        edges = [edge for edge in edges if id(edge) in kept_ids]

    if OUTPUT_SINGLE_CHILD_REPAIR and edges:
        best_by_source: dict[int, dict[str, object]] = {}

        for edge in edges:
            source_id = int(edge['source_id'])
            prev = best_by_source.get(source_id)

            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(edge) for edge in best_by_source.values()}
        stats['dropped_multi_child_edges'] = sum((1 for edge in edges if id(edge) not in kept_ids))
        edges = [edge for edge in edges if id(edge) in kept_ids]
    print(f'[{dataset}] after edge-filter+motion-relink: {len(nodes_by_id)} nodes, {len(edges)} edges')
    repair_frame_cache: dict[int, np.ndarray] = {}
    deepcenter_heatmap_cache: dict[tuple[str, int], np.ndarray] = {}
    nodes_by_id, edges = close_single_frame_gaps(nodes_by_id, edges, stats, dataset = dataset, deepcenter_bundle = deepcenter_bundle, frame_cache = repair_frame_cache, deepcenter_cache = deepcenter_heatmap_cache)
    nodes_by_id, edges = recover_strict_gap2(nodes_by_id, edges, stats, dataset = dataset)
    print(f'[{dataset}] after gap-closing (single-frame + gap2): {len(nodes_by_id)} nodes, {len(edges)} edges')
    edges = add_safe_divisions_postlink(nodes_by_id, edges, stats, dataset = dataset, deepcenter_bundle = deepcenter_bundle, frame_cache = repair_frame_cache, deepcenter_cache = deepcenter_heatmap_cache)
    _geo_cands = stats['safe_division_geometric_candidates']
    _post_veto_cands = stats['safe_division_candidates']
    _rejected_by_dc = _geo_cands - _post_veto_cands
    print(f"[{dataset}] after safe-division repair: {len(nodes_by_id)} nodes, {len(edges)} edges (geometric_candidates = {_geo_cands}, deepcenter_rejected = {_rejected_by_dc}, post_veto_candidates = {_post_veto_cands}, added = {stats['safe_divisions_added']}, cap_skipped = {stats['safe_division_skipped_cap']}, mutual_nn_rejected = {stats['safe_division_mutual_nn_rejected']}, divergence_rejected = {stats['safe_division_divergence_rejected']})")

    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:
        by_source: dict[int, list[dict[str, object]]] = {}

        for edge in edges:
            by_source.setdefault(int(edge['source_id']), []).append(edge)
        filtered: list[dict[str, object]] = []

        for source_id, source_edges in by_source.items():
            if len(source_edges) <= 1:
                filtered.extend(source_edges)
                continue
            ranked = sorted(source_edges, key = edge_sort_key, reverse = True)
            source = nodes_by_id[source_id]
            top1 = ranked[0]
            top2 = ranked[1]
            d1 = float(top1['distance_um'])
            d2 = float(top2['distance_um'])
            sister = edge_distance_um(nodes_by_id[int(top1['target_id'])], nodes_by_id[int(top2['target_id'])])
            valid_division = max(d1, d2) <= DIV_PARENT_MAX_UM and sister <= DIV_SISTER_MAX_UM and (int(nodes_by_id[int(top1['target_id'])]['t']) == int(source['t']) + 1) and (int(nodes_by_id[int(top2['target_id'])]['t']) == int(source['t']) + 1)

            if valid_division:
                filtered.extend([top1, top2])
                stats['dropped_division_edges'] += max(0, len(ranked) - 2)
            elif DIV_DROP_TO_SINGLE_IF_BAD:
                filtered.append(top1)
                stats['dropped_division_edges'] += len(ranked) - 1
            else:
                filtered.extend(ranked)
        edges = filtered

    if OUTPUT_PRUNE_ISOLATED:
        incident = {int(edge['source_id']) for edge in edges} | {int(edge['target_id']) for edge in edges}

        if incident:
            kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in incident}
            stats['pruned_isolated_nodes'] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [edge for edge in edges if int(edge['source_id']) in nodes_by_id and int(edge['target_id']) in nodes_by_id]
    print(f'[{dataset}] after division-geometry-filter+prune-isolated: {len(nodes_by_id)} nodes, {len(edges)} edges')
    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)
    print(f"[{dataset}] after short-track filtering: {len(nodes_by_id)} nodes, {len(edges)} edges (components_removed = {stats['short_track_components_removed']})")
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)
    print(f'[{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges')
    return (nodes_by_id, edges, stats)

DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()

# Write one complete test submission using the active post-processing configuration
def write_test_submission(tag: str = 'base') -> None:
    geffs = sorted((REPO_DIR / 'predictions').glob(f'*/{METHOD}/split_0/*.geff'))
    print(f'Found {len(geffs)} prediction graphs')
    found_stems = [path.stem for path in geffs]
    expected_stems = set(test_stems)
    found_set = set(found_stems)
    duplicate_stems = sorted((stem for stem, count in Counter(found_stems).items() if count > 1))
    missing = sorted(expected_stems - found_set)
    extra = sorted(found_set - expected_stems)

    if len(geffs) != len(test_stems) or duplicate_stems or missing or extra:
        raise RuntimeError(f'Expected exactly {len(test_stems)} graphs. Missing: {missing[:10]} | Extra: {extra[:10]} | Duplicates: {duplicate_stems[:10]}')

    stats_rows: list[dict[str, object]] = []
    seen_datasets: set[str] = set()
    row_id = 0
    total_nodes = 0
    total_edges = 0
    submission_temp_path = SUBMISSION_PATH.with_name(SUBMISSION_PATH.name + '.tmp')
    _unlink_resume_file(submission_temp_path)

    try:
        with submission_temp_path.open('w', newline = '') as f:
            writer = csv.DictWriter(f, fieldnames = CSV_COLUMNS)
            writer.writeheader()

            for geff_path in geffs:
                dataset = geff_path.stem
                seen_datasets.add(dataset)
                graph = graph_from_geff(geff_path)
                nodes_by_id: dict[int, dict[str, object]] = {}

                for row in graph.node_attrs().iter_rows(named = True):
                    node_id = int(row['node_id'])
                    nodes_by_id[node_id] = {'node_id': node_id, 't': int(row['t']), 'z': float(row['z']), 'y': float(row['y']), 'x': float(row['x'])}
                raw_edges: list[dict[str, object]] = []

                for row in graph.edge_attrs().iter_rows(named = True):
                    edge_prob = row.get('edge_prob') if hasattr(row, 'get') else None
                    raw_edges.append({'source_id': int(row['source_id']), 'target_id': int(row['target_id']), 'edge_prob': None if edge_prob is None else float(edge_prob)})
                raw_node_count = len(nodes_by_id)
                nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset = dataset, deepcenter_bundle = DEEPCENTER_VETO_DETECTOR)

                if not nodes_by_id:
                    raise AssertionError(f'{dataset}: post-processing removed every node')

                for node_id in sorted(nodes_by_id):
                    node = nodes_by_id[node_id]
                    writer.writerow({'id': row_id, 'dataset': dataset, 'row_type': 'node', 'node_id': int(node['node_id']), 't': int(node['t']), 'z': max(0, int(round(float(node['z'])))), 'y': max(0, int(round(float(node['y'])))), 'x': max(0, int(round(float(node['x'])))), 'source_id': -1, 'target_id': -1})
                    row_id += 1
                division_sources: dict[int, int] = {}

                for edge in edges:
                    source_id = int(edge['source_id'])
                    target_id = int(edge['target_id'])

                    if source_id not in nodes_by_id or target_id not in nodes_by_id:
                        raise AssertionError(f'{dataset}: dangling edge after filtering')
                    writer.writerow({'id': row_id, 'dataset': dataset, 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': source_id, 'target_id': target_id})
                    row_id += 1
                    division_sources[source_id] = division_sources.get(source_id, 0) + 1
                node_count = len(nodes_by_id)
                edge_count = len(edges)
                total_nodes += node_count
                total_edges += edge_count
                stats_rows.append({'dataset': dataset, 'raw_nodes': raw_node_count, 'nodes': node_count, 'raw_edges': filter_stats['raw_edges'], 'edges': edge_count, 'division_like_sources': sum((1 for count in division_sources.values() if count >= 2)), 'edge_to_node_ratio': edge_count / max(node_count, 1), 'gap_added_nodes_frac': filter_stats.get('gap_added_nodes', 0) / max(raw_node_count, 1), **filter_stats})

        expected_datasets = set(test_stems)
        missing_datasets = sorted(expected_datasets - seen_datasets)
        extra_datasets = sorted(seen_datasets - expected_datasets)

        if missing_datasets or extra_datasets:
            raise AssertionError({'missing': missing_datasets[:10], 'extra': extra_datasets[:10]})

        assert row_id == total_nodes + total_edges, 'Internal row counter mismatch'
        assert total_nodes > 0, 'No node rows produced'
        header = submission_temp_path.open().readline().strip().split(',')
        assert header == CSV_COLUMNS, f'Bad CSV header: {header}'
        submission_temp_path.replace(SUBMISSION_PATH)
    except Exception:
        _unlink_resume_file(submission_temp_path)
        raise

    stats = pd.DataFrame(stats_rows).sort_values('dataset').reset_index(drop = True)
    stats['predict_minutes_total'] = predict_seconds / 60.0
    stats['experiment_tag'] = EXPERIMENT_TAG
    stats.to_csv(RUN_STATS_PATH, index = False)
    print(f'Wrote {SUBMISSION_PATH} with {row_id:,} rows')
    print(f'Node rows: {total_nodes:,} | edge rows: {total_edges:,}')
    print(f'Wrote {RUN_STATS_PATH}')
    display(pd.read_csv(SUBMISSION_PATH, nrows = 8))

_base_postprocess_signature = _postprocess_resume_signature()
_base_submission_state = _read_resume_json(BASE_SUBMISSION_STATE_PATH)
_final_submission_state_before_base = _read_resume_json(FINAL_SUBMISSION_STATE_PATH)
_current_submission_sha = _resume_file_sha256(SUBMISSION_PATH) if SUBMISSION_PATH.is_file() else None
_base_submission_ready = bool(_base_submission_state and _base_submission_state.get('resume_schema') == RESUME_SCHEMA_VERSION and _base_submission_state.get('status') == 'complete' and _base_submission_state.get('test_prediction_signature') == _test_prediction_signature and _base_submission_state.get('postprocess_signature') == _base_postprocess_signature and _base_submission_state.get('sha256') == _current_submission_sha)
_final_submission_ready_before_base = bool(_final_submission_state_before_base and _final_submission_state_before_base.get('resume_schema') == RESUME_SCHEMA_VERSION and _final_submission_state_before_base.get('status') == 'complete' and _final_submission_state_before_base.get('test_prediction_signature') == _test_prediction_signature and _final_submission_state_before_base.get('base_postprocess_signature') == _base_postprocess_signature and _final_submission_state_before_base.get('sha256') == _current_submission_sha)

if _base_submission_ready or _final_submission_ready_before_base:
    _replay_stage_output(BASE_SUBMISSION_LOG_PATH)
    _base_preview_records = (_base_submission_state or {}).get('preview_records', [])

    if _base_preview_records:
        display(pd.DataFrame(_base_preview_records, columns = CSV_COLUMNS))
else:
    with _capture_stage_output(BASE_SUBMISSION_LOG_PATH):
        write_test_submission('base')
    _base_preview = pd.read_csv(SUBMISSION_PATH, nrows = 8)
    _write_resume_json(BASE_SUBMISSION_STATE_PATH, {'resume_schema': RESUME_SCHEMA_VERSION, 'status': 'complete', 'test_prediction_signature': _test_prediction_signature, 'postprocess_signature': _base_postprocess_signature, 'sha256': _resume_file_sha256(SUBMISSION_PATH), 'rows': _resume_csv_row_count(SUBMISSION_PATH), 'preview_records': _base_preview.to_dict(orient = 'records')})

import hashlib
import json
from pathlib import Path
import pandas as pd
import torch

# Retain the held-out validator used by the 0.946 pipeline
TRAIN_DIR = COMP_DIR / 'train'
VALIDATOR_ENABLE = os.environ.get('BIOHUB_VALIDATOR_ENABLE', '1') != '0'
VALIDATOR_N_PER_TYPE = int(os.environ.get('BIOHUB_VALIDATOR_N_PER_TYPE', '2'))
VALIDATOR_MATCH_RADIUS_UM = float(os.environ.get('BIOHUB_VALIDATOR_MATCH_RADIUS_UM', '7.0'))
VALIDATOR_NODE_COUNT_PENALTY_A = float(os.environ.get('BIOHUB_VALIDATOR_NODE_COUNT_PENALTY_A', '0.1'))
VALIDATOR_DIVISION_WEIGHT = float(os.environ.get('BIOHUB_VALIDATOR_DIVISION_WEIGHT', '0.1'))
VALIDATOR_STATS_PATH = WORKING_DIR / 'validator_results.csv'
val_stems: list[str] = []
if VALIDATOR_ENABLE and TRAIN_DIR.exists():
    train_stems_all = sorted((p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith('.zarr')))
    test_stem_set = set(test_stems)
    overlap = [s for s in train_stems_all if s in test_stem_set]
    if overlap:
        pass
    candidates = [s for s in train_stems_all if s not in test_stem_set]

    # Check whether a held-out ground-truth graph contains a division event
    def _stem_has_gt_division(stem: str) -> bool:
        gt_path = TRAIN_DIR / f'{stem}.geff'
        try:
            graph = graph_from_geff(gt_path)
        except Exception:
            return False
        out_degree: dict[int, int] = {}
        for row in graph.edge_attrs().iter_rows(named = True):
            s = int(row['source_id'])
            out_degree[s] = out_degree.get(s, 0) + 1
        return any((d >= 2 for d in out_degree.values()))
    by_prefix: dict[str, list[str]] = {}
    for s in candidates:
        by_prefix.setdefault(s.split('_')[0], []).append(s)
    division_flags: dict[str, bool] = {}
    for prefix, stems in by_prefix.items():
        for s in stems:
            division_flags[s] = _stem_has_gt_division(s)
    for prefix, stems in sorted(by_prefix.items()):
        ranked = sorted(stems, key = lambda s: (not division_flags[s], s))
        val_stems.extend(ranked[:VALIDATOR_N_PER_TYPE])
    n_division_selected = sum((1 for s in val_stems if division_flags[s]))
elif VALIDATOR_ENABLE:
    pass

# Merge validator prediction shards and verify complete held-out coverage
def _merge_validator_shards(worker_count: int, stems: list[str], method_prefix: str) -> Path:
    import shutil as _shutil
    shard_dirs: list[Path] = []
    seen: set[str] = set()
    expected_all = set(stems)
    for shard_index in range(worker_count):
        shard_method = f'{method_prefix}_gpu{shard_index}'
        shard_dir = _prediction_dir_for_method(shard_method)
        expected = set(stems[shard_index::worker_count])
        found = {p.stem for p in sorted(shard_dir.glob('*.geff'))}
        if found != expected:
            raise RuntimeError(f'VALIDATOR shard {shard_index} output mismatch: missing={sorted(expected - found)}, extra={sorted(found - expected)}')
        overlap_ds = seen & found
        if overlap_ds:
            raise RuntimeError(f'VALIDATOR: duplicate datasets across shards: {sorted(overlap_ds)}')
        seen.update(found)
        shard_dirs.append(shard_dir)
    if seen != expected_all:
        raise RuntimeError(f'VALIDATOR: merged shards do not cover the held-out set: missing={sorted(expected_all - seen)}, extra={sorted(seen - expected_all)}')
    username_roots = {shard_dir.parents[1] for shard_dir in shard_dirs}
    if len(username_roots) != 1:
        raise RuntimeError(f'VALIDATOR: shards used inconsistent prediction roots: {username_roots}')
    final_root = next(iter(username_roots)) / method_prefix
    final_dir = final_root / 'split_0'
    staging_dir = final_root / 'split_0_val_staging'
    if staging_dir.exists():
        _shutil.rmtree(staging_dir) if staging_dir.is_dir() else staging_dir.unlink()
    staging_dir.mkdir(parents = True, exist_ok = False)
    for shard_dir in shard_dirs:
        for source in sorted(shard_dir.glob('*.geff')):
            destination = staging_dir / source.name
            if destination.exists():
                raise RuntimeError(f'VALIDATOR: refusing to overwrite duplicate output: {destination}')
            _shutil.move(str(source), str(destination))
    merged = {p.stem for p in staging_dir.glob('*.geff')}
    if merged != expected_all:
        raise RuntimeError(f'VALIDATOR: staged directory failed verification: missing={sorted(expected_all - merged)}, extra={sorted(merged - expected_all)}')
    if final_dir.exists():
        _shutil.rmtree(final_dir) if final_dir.is_dir() else final_dir.unlink()
    staging_dir.rename(final_dir)
    for shard_dir in shard_dirs:
        _shutil.rmtree(shard_dir.parent)
    return final_dir

# Run held-out inference through the same model path used for test prediction
predict_val_seconds = None
if VALIDATOR_ENABLE and val_stems:
    val_splits_path = REPO_DIR / 'kaggle_val_splits.json'
    val_splits_path.write_text(json.dumps([{'split': 0, 'train': [], 'test': val_stems}], indent = 2))
    val_method_prefix = f'{METHOD}_val'
    predict_val_cmd = [sys.executable, 'scripts/predict_unet_transformer.py', '--data-dir', str(TRAIN_DIR), '--splits', str(val_splits_path.name), '--split', '0', '--weights', WEIGHTS_RELATIVE, '--unet-batch-size', str(UNET_BATCH_SIZE), '--det-threshold', str(DET_THRESHOLD), '--ilp-edge-weight', str(ILP_EDGE_WEIGHT), '--ilp-appearance-weight', str(ILP_APPEARANCE_WEIGHT), '--ilp-disappearance-weight', str(ILP_DISAPPEARANCE_WEIGHT), '--ilp-division-weight', str(ILP_DIVISION_WEIGHT)]

    if USE_ILP:
        predict_val_cmd.append('--use-ilp')
    _validator_prediction_signature = _resume_signature({'resume_schema': RESUME_SCHEMA_VERSION, 'test_prediction_signature': _test_prediction_signature, 'val_stems': val_stems, 'method': val_method_prefix})
    _validator_prediction_state = _read_resume_json(VALIDATOR_PREDICTION_STATE_PATH)
    _validator_prediction_ready = bool(_validator_prediction_state and _validator_prediction_state.get('resume_schema') == RESUME_SCHEMA_VERSION and _validator_prediction_state.get('status') == 'complete' and _validator_prediction_state.get('signature') == _validator_prediction_signature and _resume_prediction_complete(val_method_prefix, val_stems))

    if _validator_prediction_ready:
        predict_val_seconds = float(_validator_prediction_state.get('predict_val_seconds', 0.0))

        if VALIDATOR_PREDICTION_LOG_PATH.is_file():
            _replay_stage_output(VALIDATOR_PREDICTION_LOG_PATH)
        else:
            print('VALIDATOR prediction minutes:', round(predict_val_seconds / 60.0, 2))
    else:
        _remove_prediction_methods(val_method_prefix)

        for _guard_old_log in WORKING_DIR.glob('retention_guard_val_*.jsonl'):
            _guard_old_log.unlink()

        for _coordinate_old_log in WORKING_DIR.glob('detector_coordinates_*_val_*.jsonl'):
            _coordinate_old_log.unlink()

        for _dependent_path in [VALIDATOR_BASE_STATE_PATH, PPSWEEP_STATE_PATH, FINAL_SUBMISSION_STATE_PATH]:
            _unlink_resume_file(_dependent_path)

        for _dependent_log in [VALIDATOR_BASE_LOG_PATH, PPSWEEP_LOG_PATH, FINAL_SUBMISSION_LOG_PATH]:
            _unlink_resume_file(_dependent_log)

        with _capture_stage_output(VALIDATOR_PREDICTION_LOG_PATH):
            _val_start = time.time()
            val_worker_count = min(2, _torch.cuda.device_count(), len(val_stems))

            if val_worker_count >= 2:
                cuda_tokens = _visible_cuda_tokens(val_worker_count)
                val_processes: dict[int, subprocess.Popen] = {}
                val_commands: dict[int, list[str]] = {}

                for shard_index in range(val_worker_count):
                    shard_cmd = [*predict_val_cmd, '--method', f'{val_method_prefix}_gpu{shard_index}', '--slice', f'{shard_index}::{val_worker_count}']
                    shard_env = {**os.environ, 'PYTHONPATH': 'src'}
                    shard_env['CUDA_VISIBLE_DEVICES'] = cuda_tokens[shard_index]
                    shard_env['BIOHUB_GPU_SHARD'] = f'val_{shard_index}_{val_worker_count}'
                    val_commands[shard_index] = shard_cmd
                    val_processes[shard_index] = _popen_streamed(shard_cmd, cwd = REPO_DIR, env = shard_env)
                _wait_for_prediction_shards(val_processes, val_commands)
                _merge_validator_shards(val_worker_count, val_stems, val_method_prefix)
            else:
                val_env = {**os.environ, 'PYTHONPATH': 'src'}
                val_env['BIOHUB_GPU_SHARD'] = 'val_single'
                _run_subprocess_streamed([*predict_val_cmd, '--method', val_method_prefix], cwd = REPO_DIR, env = val_env)
            predict_val_seconds = time.time() - _val_start
            print('VALIDATOR prediction minutes:', round(predict_val_seconds / 60.0, 2))

        if not _resume_prediction_complete(val_method_prefix, val_stems):
            raise RuntimeError('Validator prediction stage finished without a complete prediction cache')
        _write_resume_json(VALIDATOR_PREDICTION_STATE_PATH, {'resume_schema': RESUME_SCHEMA_VERSION, 'status': 'complete', 'signature': _validator_prediction_signature, 'predict_val_seconds': predict_val_seconds, 'val_stems': val_stems, 'method': val_method_prefix})
else:
    _validator_prediction_signature = _resume_signature({'resume_schema': RESUME_SCHEMA_VERSION, 'disabled': True, 'val_stems': val_stems})

from scipy.optimize import linear_sum_assignment

# Match predicted and ground-truth nodes with one-to-one spatial assignment
def match_nodes_bipartite(pred_nodes: dict, gt_nodes: dict, max_dist: float = 7.0):
    pred_by_t: dict[int, list[int]] = {}
    for pid, (t, *_r) in pred_nodes.items():
        pred_by_t.setdefault(int(t), []).append(pid)
    gt_by_t: dict[int, list[int]] = {}
    for gid, (t, *_r) in gt_nodes.items():
        gt_by_t.setdefault(int(t), []).append(gid)
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for t, p_ids in pred_by_t.items():
        g_ids = gt_by_t.get(t, [])
        if not g_ids:
            continue
        voxel_scale = np.array(VOXEL_SCALE_UM, dtype = float)
        p_pos = np.array([pred_nodes[p][1:] for p in p_ids], dtype = float) * voxel_scale
        g_pos = np.array([gt_nodes[g][1:] for g in g_ids], dtype = float) * voxel_scale
        diff = p_pos[:, None, :] - g_pos[None, :, :]
        cost = np.sqrt((diff ** 2).sum(axis = -1))
        BIG = 1000000.0
        cost_gated = np.where(cost <= max_dist, cost, BIG)
        row_ind, col_ind = linear_sum_assignment(cost_gated)
        for r, c in zip(row_ind, col_ind):
            if cost_gated[r, c] >= BIG:
                continue
            pred_to_gt[p_ids[r]] = g_ids[c]
            gt_to_pred[g_ids[c]] = p_ids[r]
    return (pred_to_gt, gt_to_pred)

# Compute edge true positives, false positives, and false negatives after node matching
def compute_edge_confusion(pred_edges, gt_edges, pred_to_gt, gt_to_pred):
    gt_edge_set = set(gt_edges)
    gt_outgoing: dict[int, set[int]] = {}
    gt_incoming_source: dict[int, int] = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)
        gt_incoming_source[t] = s
    tp = 0
    fp = 0
    matched_gt_edges = set()
    for s, t in pred_edges:
        ms = pred_to_gt.get(s)
        mt = pred_to_gt.get(t)
        is_tp = ms is not None and mt is not None and (mt in gt_outgoing.get(ms, ()))
        if is_tp:
            tp += 1
            matched_gt_edges.add((ms, mt))
            continue
        is_fp = mt is not None and mt in gt_incoming_source or (ms is not None and bool(gt_outgoing.get(ms)))
        if is_fp:
            fp += 1
    fn = len(gt_edge_set - matched_gt_edges)
    return (tp, fp, fn)

# Compute the edge jaccard component of the validator metric
def edge_jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom else 0.0

# Apply the node-count penalty to the edge jaccard score
def adjusted_jaccard(jaccard: float, t_pred: int, t_true, a: float = 0.1) -> float:
    if not t_true or t_true <= 0:
        return jaccard
    return max(0.0, jaccard * (1.0 - a * (t_pred - t_true) / t_true))

# Build weakly connected components for patched division matching
def weakly_connected_components(node_ids, edges):
    parent = {n: n for n in node_ids}

    # Find the current disjoint-set representative
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Merge two disjoint-set components
    def union(a, b):
        ra, rb = (find(a), find(b))
        if ra != rb:
            parent[ra] = rb
    for s, t in edges:
        if s in parent and t in parent:
            union(s, t)
    return {n: find(n) for n in node_ids}

# Compute division true positives, false positives, and false negatives
def compute_division_confusion(pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred):
    gt_out: dict[int, set[int]] = {}
    gt_in: dict[int, int] = {}
    for s, t in gt_edges:
        gt_out.setdefault(s, set()).add(t)
        gt_in[t] = s
    pred_out: dict[int, set[int]] = {}
    for s, t in pred_edges:
        pred_out.setdefault(s, set()).add(t)
    pred_node_ids = list(pred_nodes.keys())
    pred_edge_list = list(pred_edges)
    components = weakly_connected_components(pred_node_ids, pred_edge_list)
    fork_components = {components[n] for n, outs in pred_out.items() if len(outs) >= 2 and n in components}
    gt_division_sources = [s for s, outs in gt_out.items() if len(outs) >= 2]

    # Collect lineage descendants used by patched division matching
    def lineage_descendants(root_child: int) -> set[int]:
        seen = {root_child}
        stack = [root_child]
        while stack:
            cur = stack.pop()
            for nxt in gt_out.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen
    tp = 0
    fn = 0
    tp_gt_sources: set[int] = set()
    for gsrc in gt_division_sources:
        children = sorted(gt_out[gsrc])
        if len(children) < 2:
            continue
        anchor_candidates = [gsrc]
        if gsrc in gt_in:
            anchor_candidates.append(gt_in[gsrc])
        anchor_pred_nodes = [gt_to_pred[a] for a in anchor_candidates if a in gt_to_pred]
        lineage_hit_components: list[set[int]] = []
        ok = True
        for child in children[:2]:
            lineage = lineage_descendants(child)
            hit_comp_ids = {components[p_id] for gt_id in lineage if (p_id := gt_to_pred.get(gt_id)) is not None and p_id in components}
            if not hit_comp_ids:
                ok = False
                break
            lineage_hit_components.append(hit_comp_ids)
        if not ok or not anchor_pred_nodes:
            fn += 1
            continue
        anchor_comp_ids = {components[p] for p in anchor_pred_nodes if p in components}
        if not anchor_comp_ids:
            fn += 1
            continue
        found = any((comp_id in lineage_hit_components[0] and comp_id in lineage_hit_components[1] and (comp_id in fork_components) for comp_id in anchor_comp_ids))
        if found:
            tp += 1
            tp_gt_sources.add(gsrc)
        else:
            fn += 1
    fp = 0
    for n, outs in pred_out.items():
        if len(outs) < 2:
            continue
        g = pred_to_gt.get(n)
        if g is None or g not in gt_out or g in tp_gt_sources:
            continue
        fp += 1
    return (tp, fp, fn)

# Decompose validator errors into the official metric components
def decompose_errors(pred_nodes, gt_nodes, pred_edges, gt_edges, pred_to_gt, gt_to_pred):
    gt_edge_set = set(gt_edges)
    pred_edge_set = set(pred_edges)
    gt_outgoing: dict[int, set[int]] = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)
    missed_gt_nodes = sum((1 for g in gt_nodes if g not in gt_to_pred))
    spurious_pred_nodes = sum((1 for p in pred_nodes if p not in pred_to_gt))
    recovered = fragmented = lost_to_detection = 0
    for gs, gtid in gt_edge_set:
        ps, pt = (gt_to_pred.get(gs), gt_to_pred.get(gtid))
        if ps is None or pt is None:
            lost_to_detection += 1
        elif (ps, pt) in pred_edge_set:
            recovered += 1
        else:
            fragmented += 1
    wrong_association = 0
    for ps, pt in pred_edge_set:
        ms, mt = (pred_to_gt.get(ps), pred_to_gt.get(pt))
        if ms is not None and mt is not None and (mt not in gt_outgoing.get(ms, ())):
            wrong_association += 1
    return {'missed_gt_nodes': missed_gt_nodes, 'spurious_pred_nodes': spurious_pred_nodes, 'edges_recovered': recovered, 'edges_fragmented': fragmented, 'edges_lost_to_detection': lost_to_detection, 'wrong_association_edges': wrong_association}

# Read a nested metadata key from the ground-truth geff payload
def _find_key_recursive(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None

# Read the estimated true node count used by the metric penalty
def read_estimated_true_node_count(geff_path: Path):
    for candidate in (geff_path / 'zarr.json', geff_path / '.zattrs'):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        found = _find_key_recursive(payload, 'estimated_number_of_nodes')
        if found is not None:
            try:
                return float(found)
            except (TypeError, ValueError):
                continue
    return None

# Convert a geff graph into plain node and edge structures
def graph_to_plain(graph):
    nodes: dict[int, tuple] = {}
    for row in graph.node_attrs().iter_rows(named = True):
        node_id = int(row['node_id'])
        nodes[node_id] = (int(row['t']), float(row['z']), float(row['y']), float(row['x']))
    edges: list[tuple[int, int]] = []
    for row in graph.edge_attrs().iter_rows(named = True):
        edges.append((int(row['source_id']), int(row['target_id'])))
    return (nodes, edges)

# Convert processed node dictionaries into validator node tuples
def nodes_by_id_to_plain(nodes_by_id):
    return {nid: (int(n['t']), float(n['z']), float(n['y']), float(n['x'])) for nid, n in nodes_by_id.items()}

# Score one held-out sample with the official metric components
def score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true):
    p2g, g2p = match_nodes_bipartite(pred_nodes_plain, gt_nodes_plain, max_dist = VALIDATOR_MATCH_RADIUS_UM)
    tp, fp, fn = compute_edge_confusion(pred_edges_plain, gt_edges_plain, p2g, g2p)
    jac = edge_jaccard(tp, fp, fn)
    t_pred = len(pred_nodes_plain)
    adj = adjusted_jaccard(jac, t_pred, t_true, a = VALIDATOR_NODE_COUNT_PENALTY_A)
    div_tp, div_fp, div_fn = compute_division_confusion(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, p2g, g2p)
    errors = decompose_errors(pred_nodes_plain, gt_nodes_plain, pred_edges_plain, gt_edges_plain, p2g, g2p)
    div_jac = edge_jaccard(div_tp, div_fp, div_fn)
    row = {'edge_tp': tp, 'edge_fp': fp, 'edge_fn': fn, 'edge_jaccard': jac, 't_pred': t_pred, 't_true': t_true, 'adjusted_edge_jaccard': adj, 'div_tp': div_tp, 'div_fp': div_fp, 'div_fn': div_fn, 'div_jaccard': div_jac, 'weight': tp + fp + fn}
    row.update(errors)
    return row

# Aggregate held-out sample scores with the official competition formula
def aggregate_official(sample_rows):
    total_w = sum((r['weight'] for r in sample_rows)) or 1
    weighted_adj = sum((r['adjusted_edge_jaccard'] * r['weight'] for r in sample_rows)) / total_w
    div_tp = sum((r['div_tp'] for r in sample_rows))
    div_fp = sum((r['div_fp'] for r in sample_rows))
    div_fn = sum((r['div_fn'] for r in sample_rows))
    div_jac = edge_jaccard(div_tp, div_fp, div_fn)
    return {'adjusted_edge_jaccard': weighted_adj, 'division_jaccard': div_jac, 'proxy_score': weighted_adj + VALIDATOR_DIVISION_WEIGHT * div_jac, 'div_tp': div_tp, 'div_fp': div_fp, 'div_fn': div_fn, 'missed_gt_nodes': sum((r['missed_gt_nodes'] for r in sample_rows)), 'spurious_pred_nodes': sum((r['spurious_pred_nodes'] for r in sample_rows)), 'edges_recovered': sum((r['edges_recovered'] for r in sample_rows)), 'edges_fragmented': sum((r['edges_fragmented'] for r in sample_rows)), 'edges_lost_to_detection': sum((r['edges_lost_to_detection'] for r in sample_rows)), 'wrong_association_edges': sum((r['wrong_association_edges'] for r in sample_rows))}

import copy as _copy

PP_SWEEP_KEYS = ['SAFE_DIV_MAX_UM', 'SAFE_DIV_SISTER_MAX_UM', 'SAFE_DIV_DIVERGE_UM', 'SAFE_DIV_SISTER_SYMMETRY_TAU', 'SAFE_DIV_EXISTING_CHILD_MAX_UM', 'SAFE_DIV_FRAME_FRAC_CAP', 'SAFE_DIV_GLOBAL_FRAC_CAP', 'DEEPCENTER_SAFE_DIV_THRESHOLD', 'DEEPCENTER_GAP_THRESHOLD', 'GAP_CLOSE_UM', 'OUTPUT_MIN_TRACK_LEN', 'SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB', 'MOTION_RELINK_TIGHT_UM', 'MOTION_RELINK_RELAXED_UM', 'GAP2_MAX_STEP_UM', 'GAP2_MAX_TOTAL_UM', 'MOTION_RELINK_LEARNED_BONUS', 'MOTION_RELINK_VELOCITY_WEIGHT', 'GAP_CLOSE_REUSE_UM', 'OUTPUT_EDGE_MAX_UM']
PP_BASE_CONFIG = {key: globals()[key] for key in PP_SWEEP_KEYS}
for key in PP_SWEEP_KEYS:
    pass

# Apply one post-process candidate while preserving the current configuration
def pp_apply(config: dict) -> dict:
    saved = {key: globals()[key] for key in config}
    for key, value in config.items():
        if key not in PP_SWEEP_KEYS:
            raise KeyError(f'{key} is not a sweepable post-process constant')
        globals()[key] = type(PP_BASE_CONFIG[key])(value)
    return saved

# Restore the post-process configuration after candidate evaluation
def pp_restore(saved: dict) -> None:
    for key, value in saved.items():
        globals()[key] = value
VAL_RAW_GRAPHS: dict[str, tuple[dict, list]] = {}
VAL_GT: dict[str, tuple[list, list, object]] = {}
if VALIDATOR_ENABLE and val_stems:
    val_pred_paths = {stem: found for stem in val_stems if (found := next((REPO_DIR / 'predictions').rglob(f'{stem}.geff'), None)) is not None}
    missing = [s for s in val_stems if s not in val_pred_paths]
    if missing:
        raise RuntimeError(f'VALIDATOR: no prediction .geff for {missing}')
    for stem in val_stems:
        gt_path = TRAIN_DIR / f'{stem}.geff'
        if not gt_path.exists():
            raise RuntimeError(f'VALIDATOR: missing GT {gt_path}')
        gt_graph = graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)
        t_true = read_estimated_true_node_count(gt_path)
        if t_true is None:
            raise RuntimeError(f'VALIDATOR: estimated_number_of_nodes missing for {stem}')
        VAL_GT[stem] = (gt_nodes_plain, gt_edges_plain, t_true)
        pred_graph = graph_from_geff(val_pred_paths[stem])
        raw_nodes_by_id: dict[int, dict[str, object]] = {}
        for row in pred_graph.node_attrs().iter_rows(named = True):
            node_id = int(row['node_id'])
            raw_nodes_by_id[node_id] = {'node_id': node_id, 't': int(row['t']), 'z': float(row['z']), 'y': float(row['y']), 'x': float(row['x'])}
        raw_edges = []
        for row in pred_graph.edge_attrs().iter_rows(named = True):
            edge_prob = row.get('edge_prob') if hasattr(row, 'get') else None
            raw_edges.append({'source_id': int(row['source_id']), 'target_id': int(row['target_id']), 'edge_prob': None if edge_prob is None else float(edge_prob)})
        VAL_RAW_GRAPHS[stem] = (raw_nodes_by_id, raw_edges)

# Run post-processing on cached validator graphs and score one candidate
def score_validator_config(config: dict, label: str, verbose: bool = False) -> tuple[dict, list]:
    saved = pp_apply(config)
    _real_test_dir = TEST_DIR
    globals()['TEST_DIR'] = TRAIN_DIR
    rows = []
    t0 = time.time()
    try:
        for stem in val_stems:
            raw_nodes_by_id, raw_edges = VAL_RAW_GRAPHS[stem]
            nodes_copy = _copy.deepcopy(raw_nodes_by_id)
            edges_copy = _copy.deepcopy(raw_edges)
            processed_nodes, processed_edges, _stage_stats = filter_output_graph(nodes_copy, edges_copy, dataset = stem, deepcenter_bundle = globals().get('DEEPCENTER_VETO_DETECTOR'))
            gt_nodes_plain, gt_edges_plain, t_true = VAL_GT[stem]
            pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)
            pred_edges_plain = [(int(e['source_id']), int(e['target_id'])) for e in processed_edges]
            row = score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true)
            row['stem'] = stem
            row['config'] = label
            row['safe_divisions_added'] = _stage_stats.get('safe_divisions_added', 0)
            rows.append(row)
    finally:
        globals()['TEST_DIR'] = _real_test_dir
        pp_restore(saved)
    summary = aggregate_official(rows)
    summary['n_samples'] = len(rows)
    summary['config'] = label
    summary['seconds'] = time.time() - t0
    if verbose:
        for row in rows:
            pass
    return (summary, rows)
validator_sample_rows: list[dict[str, object]] = []
validator_summary_rows: list[dict[str, object]] = []
PP_RESULTS: dict[str, dict] = {}
if VALIDATOR_ENABLE and val_stems:
    _validator_base_signature = _resume_signature({'resume_schema': RESUME_SCHEMA_VERSION, 'validator_prediction_signature': _validator_prediction_signature, 'postprocess': PP_BASE_CONFIG, 'match_radius_um': VALIDATOR_MATCH_RADIUS_UM, 'node_count_penalty_a': VALIDATOR_NODE_COUNT_PENALTY_A, 'division_weight': VALIDATOR_DIVISION_WEIGHT})
    _validator_base_state = _read_resume_json(VALIDATOR_BASE_STATE_PATH)
    _validator_base_ready = bool(_validator_base_state and _validator_base_state.get('resume_schema') == RESUME_SCHEMA_VERSION and _validator_base_state.get('status') == 'complete' and _validator_base_state.get('signature') == _validator_base_signature)

    if _validator_base_ready:
        base_summary = dict(_validator_base_state['summary'])
        base_rows = list(_validator_base_state['rows'])
        _replay_stage_output(VALIDATOR_BASE_LOG_PATH)
    else:
        with _capture_stage_output(VALIDATOR_BASE_LOG_PATH):
            base_summary, base_rows = score_validator_config({}, 'base', verbose = True)
            print('VALIDATOR base proxy:', round(base_summary['proxy_score'], 6))
            print('VALIDATOR base adjusted edge:', round(base_summary['adjusted_edge_jaccard'], 6))
        _write_resume_json(VALIDATOR_BASE_STATE_PATH, {'resume_schema': RESUME_SCHEMA_VERSION, 'status': 'complete', 'signature': _validator_base_signature, 'summary': base_summary, 'rows': base_rows})

    PP_RESULTS['base'] = base_summary
    validator_sample_rows.extend(base_rows)
    validator_summary_rows.append(base_summary)

    with VALIDATOR_STATS_PATH.open('w', newline = '') as f:
        fieldnames = sorted({k for row in base_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()

        for row in base_rows:
            writer.writerow(row)

# Retain the narrow post-process candidate set used by the 0.946 pipeline
PP_CANDIDATES: dict[str, dict] = {'gap45': {'GAP_CLOSE_UM': 4.5}, 'tight55': {'MOTION_RELINK_TIGHT_UM': 5.5}, 'relaxed9': {'MOTION_RELINK_RELAXED_UM': 9.0}, 'bonus125': {'MOTION_RELINK_LEARNED_BONUS': 1.25}, 'gap2step40': {'GAP2_MAX_STEP_UM': 4.0}, 'reuse28': {'GAP_CLOSE_REUSE_UM': 2.8}, 'dcgap035': {'DEEPCENTER_GAP_THRESHOLD': 0.35}}
PP_SELECT_MARGIN = float(os.environ.get('BIOHUB_PPSWEEP_SELECT_MARGIN', '0.002'))
PP_MAX_ADJ_LOSS = float(os.environ.get('BIOHUB_PPSWEEP_MAX_ADJ_LOSS', '0.0005'))
PP_SWEEP_RESULTS_PATH = WORKING_DIR / 'ppsweep_results.csv'
PP_SELECTED_PATH = WORKING_DIR / 'ppsweep_selected.json'

# Select a candidate only when it clears both validator gates
selected_label = 'base'
selected_config: dict = {}
_pp_sweep_signature = _resume_signature({'resume_schema': RESUME_SCHEMA_VERSION, 'validator_prediction_signature': _validator_prediction_signature, 'validator_base': PP_RESULTS.get('base'), 'pp_base_config': PP_BASE_CONFIG, 'pp_candidates': PP_CANDIDATES, 'select_margin': PP_SELECT_MARGIN, 'max_adj_loss': PP_MAX_ADJ_LOSS})
_pp_sweep_state = _read_resume_json(PPSWEEP_STATE_PATH)
_pp_sweep_ready = bool(VALIDATOR_ENABLE and val_stems and ('base' in PP_RESULTS) and _pp_sweep_state and _pp_sweep_state.get('resume_schema') == RESUME_SCHEMA_VERSION and _pp_sweep_state.get('status') == 'complete' and _pp_sweep_state.get('signature') == _pp_sweep_signature)

if _pp_sweep_ready:
    PP_RESULTS = {str(key): dict(value) for key, value in _pp_sweep_state.get('pp_results', {}).items()}
    PP_CANDIDATES = {str(key): dict(value) for key, value in _pp_sweep_state.get('pp_candidates', PP_CANDIDATES).items()}
    selected_label = str(_pp_sweep_state.get('selected_label', 'base'))
    selected_config = dict(_pp_sweep_state.get('selected_config', {}))
    validator_sample_rows = list(_pp_sweep_state.get('validator_sample_rows', validator_sample_rows))
    _replay_stage_output(PPSWEEP_LOG_PATH)
elif VALIDATOR_ENABLE and val_stems and ('base' in PP_RESULTS):
    with _capture_stage_output(PPSWEEP_LOG_PATH):
        base_summary = PP_RESULTS['base']

        for label, config in PP_CANDIDATES.items():
            summary, rows = score_validator_config(config, label)
            PP_RESULTS[label] = summary
            print('VALIDATOR candidate:', label, 'proxy =', round(summary['proxy_score'], 6), 'adjusted edge =', round(summary['adjusted_edge_jaccard'], 6))
            validator_sample_rows.extend(rows)
        positive = [label for label, summary in PP_RESULTS.items() if label != 'base' and summary['proxy_score'] >= base_summary['proxy_score'] + 0.0005 and (summary['adjusted_edge_jaccard'] >= base_summary['adjusted_edge_jaccard'] - PP_MAX_ADJ_LOSS)]
        positive.sort(key = lambda l: PP_RESULTS[l]['proxy_score'], reverse = True)
        combo_config: dict = {}

        for label in positive:
            for key, value in PP_CANDIDATES[label].items():
                combo_config.setdefault(key, value)

        if len(positive) >= 2:
            combo_label = 'combo(' + '+'.join(positive) + ')'
            summary, rows = score_validator_config(combo_config, combo_label)
            PP_RESULTS[combo_label] = summary
            PP_CANDIDATES[combo_label] = combo_config
            validator_sample_rows.extend(rows)
        ranked = sorted(PP_RESULTS.items(), key = lambda kv: kv[1]['proxy_score'], reverse = True)

        for label, summary in ranked:
            delta = summary['proxy_score'] - base_summary['proxy_score']
        pd.DataFrame([{'config': label, **{k: v for k, v in summary.items() if k != 'config'}, 'overrides': json.dumps(PP_CANDIDATES.get(label, {}), sort_keys = True)} for label, summary in ranked]).to_csv(PP_SWEEP_RESULTS_PATH, index = False)
        best_label, best_summary = ranked[0]

        if best_label != 'base' and best_summary['proxy_score'] >= base_summary['proxy_score'] + PP_SELECT_MARGIN and (best_summary['adjusted_edge_jaccard'] >= base_summary['adjusted_edge_jaccard'] - PP_MAX_ADJ_LOSS):
            selected_label = best_label
            selected_config = dict(PP_CANDIDATES[best_label])
        print('VALIDATOR selected config:', selected_label, selected_config)
        sel = PP_RESULTS[selected_label]

        with VALIDATOR_STATS_PATH.open('w', newline = '') as f:
            fieldnames = sorted({k for row in validator_sample_rows for k in row.keys()})
            writer = csv.DictWriter(f, fieldnames = fieldnames)
            writer.writeheader()

            for row in validator_sample_rows:
                writer.writerow(row)
    _write_resume_json(PPSWEEP_STATE_PATH, {'resume_schema': RESUME_SCHEMA_VERSION, 'status': 'complete', 'signature': _pp_sweep_signature, 'pp_results': PP_RESULTS, 'pp_candidates': PP_CANDIDATES, 'selected_label': selected_label, 'selected_config': selected_config, 'validator_sample_rows': validator_sample_rows})

# Save the resolved post-process selection for reproducibility
PP_SELECTED_PATH.write_text(json.dumps({'selected': selected_label, 'overrides': selected_config, 'base_proxy': PP_RESULTS.get('base', {}).get('proxy_score'), 'selected_proxy': PP_RESULTS.get(selected_label, {}).get('proxy_score'), 'held_out_stems': list(val_stems)}, indent = 2, sort_keys = True) + '\n')

# Rewrite submission.csv only when the validator selects a post-process candidate
_final_submission_signature = _resume_signature({'resume_schema': RESUME_SCHEMA_VERSION, 'test_prediction_signature': _test_prediction_signature, 'base_postprocess_signature': _base_postprocess_signature, 'ppsweep_signature': _pp_sweep_signature, 'selected_label': selected_label, 'selected_config': selected_config})
_final_submission_state = _read_resume_json(FINAL_SUBMISSION_STATE_PATH)
_current_submission_sha = _resume_file_sha256(SUBMISSION_PATH) if SUBMISSION_PATH.is_file() else None
_final_submission_ready = bool(_final_submission_state and _final_submission_state.get('resume_schema') == RESUME_SCHEMA_VERSION and _final_submission_state.get('status') == 'complete' and _final_submission_state.get('signature') == _final_submission_signature and _final_submission_state.get('sha256') == _current_submission_sha)

if selected_config:
    print('VALIDATOR rewriting submission with config:', selected_label)

    if _final_submission_ready:
        _replay_stage_output(FINAL_SUBMISSION_LOG_PATH)
        _final_preview_records = _final_submission_state.get('preview_records', [])

        if _final_preview_records:
            display(pd.DataFrame(_final_preview_records, columns = CSV_COLUMNS))
    else:
        with _capture_stage_output(FINAL_SUBMISSION_LOG_PATH):
            _saved = pp_apply(selected_config)

            try:
                write_test_submission(selected_label)
            finally:
                pp_restore(_saved)
        _final_preview = pd.read_csv(SUBMISSION_PATH, nrows = 8)
        _write_resume_json(FINAL_SUBMISSION_STATE_PATH, {'resume_schema': RESUME_SCHEMA_VERSION, 'status': 'complete', 'signature': _final_submission_signature, 'test_prediction_signature': _test_prediction_signature, 'base_postprocess_signature': _base_postprocess_signature, 'selected_label': selected_label, 'selected_config': selected_config, 'sha256': _resume_file_sha256(SUBMISSION_PATH), 'rows': _resume_csv_row_count(SUBMISSION_PATH), 'preview_records': _final_preview.to_dict(orient = 'records')})

    for key, value in selected_config.items():
        os.environ['BIOHUB_' + key] = str(value)
else:
    if not _final_submission_ready:
        _final_preview = pd.read_csv(SUBMISSION_PATH, nrows = 8)
        _write_resume_json(FINAL_SUBMISSION_STATE_PATH, {'resume_schema': RESUME_SCHEMA_VERSION, 'status': 'complete', 'signature': _final_submission_signature, 'test_prediction_signature': _test_prediction_signature, 'base_postprocess_signature': _base_postprocess_signature, 'selected_label': selected_label, 'selected_config': selected_config, 'sha256': _resume_file_sha256(SUBMISSION_PATH), 'rows': _resume_csv_row_count(SUBMISSION_PATH), 'preview_records': _final_preview.to_dict(orient = 'records')})

# Audit the final submission after validator-driven post-process selection
_final = pd.read_csv(SUBMISSION_PATH)
assert _final.columns.tolist() == CSV_COLUMNS, _final.columns.tolist()
assert _final['id'].tolist() == list(range(len(_final))), 'row ids not contiguous'
_expected_sets = sorted((p.name[:-5] for p in TEST_DIR.iterdir() if p.name.endswith('.zarr')))
assert sorted(_final['dataset'].astype(str).unique()) == _expected_sets, 'dataset mismatch'
print('VALIDATOR final submission rows:', len(_final))
print('VALIDATOR final config:', selected_label)
for _ds, _grp in _final.groupby('dataset'):
    _n = _grp[_grp.row_type.eq('node')]
    _e = _grp[_grp.row_type.eq('edge')]
    _t = dict(zip(_n.node_id.astype(int), _n.t.astype(int)))
    assert all((_t[int(s)] + 1 == _t[int(d)] for s, d in zip(_e.source_id, _e.target_id))), f'{_ds}: bad edge time'
    _max_indegree = int(_e.target_id.value_counts().max()) if not _e.empty else 0
    _max_outdegree = int(_e.source_id.value_counts().max()) if not _e.empty else 0
    assert _max_indegree <= 1, f'{_ds}: multi-parent'
    assert _max_outdegree <= 2, f'{_ds}: out-degree > 2'
_secondary_weights_env = os.environ.get('BIOHUB_SECONDARY_WEIGHTS', '')
_secondary_ready = bool(_secondary_weights_env and Path(_secondary_weights_env).exists())
_bidir_weight = float(os.environ.get('BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT', '0'))
_dc_requested = os.environ.get('BIOHUB_USE_DEEPCENTER_VETO', '0') != '0'
_dc_bundle = globals().get('DEEPCENTER_VETO_DETECTOR')
_dc_loaded = 'DEEPCENTER_VETO_DETECTOR' in globals() and _dc_bundle is not None
_dc_path = _dc_bundle.get('path') if _dc_loaded else None
if _dc_loaded:
    pass

_guard_submission = Path('/kaggle/working/submission.csv')
_guard_columns = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']

if not _guard_submission.is_file():
    raise FileNotFoundError(_guard_submission)

_guard_frame = pd.read_csv(_guard_submission)

if _guard_frame.empty or _guard_frame.columns.tolist() != _guard_columns:
    raise RuntimeError('Retention-guard submission schema changed')

if _guard_frame['id'].tolist() != list(range(len(_guard_frame))):
    raise RuntimeError('Retention-guard row IDs are not contiguous')

if set(_guard_frame['row_type'].unique()) != {'node', 'edge'}:
    raise RuntimeError('Retention-guard row types changed')

_guard_datasets = sorted(_guard_frame['dataset'].astype(str).unique())
_guard_expected = sorted((path.name.removesuffix('.zarr') for path in TEST_DIR.iterdir() if path.name.endswith('.zarr')))

if _guard_datasets != _guard_expected:
    raise RuntimeError({'expected': _guard_expected, 'actual': _guard_datasets})

_guard_records = []
_guard_expected_set = set(_guard_expected)

for _guard_path in sorted(Path('/kaggle/working').glob('retention_guard_*.jsonl')):
    for _guard_line in _guard_path.read_text().splitlines():
        if _guard_line.strip():
            _guard_record = json.loads(_guard_line)

            if str(_guard_record.get('dataset', '')) in _guard_expected_set:
                _guard_records.append(_guard_record)

if not _guard_records:
    raise RuntimeError('No frame-retention diagnostics were produced')

_guard_keys = [(str(row['dataset']), int(row['frame'])) for row in _guard_records]

if len(_guard_keys) != len(set(_guard_keys)):
    raise RuntimeError('Duplicate frame-retention diagnostics')

if sorted(set((movie for movie, _ in _guard_keys))) != _guard_expected:
    raise RuntimeError('Frame-retention diagnostics do not cover every movie')

for _guard_record in _guard_records:
    if float(_guard_record['minimum_retention']) != 0.9 or int(_guard_record['primary_candidates']) < 0 or int(_guard_record['blended_candidates']) < 0:
        raise RuntimeError('Frame-retention diagnostic contract changed')
    _guard_expected_use_primary = bool(int(_guard_record['primary_candidates']) > 0 and float(_guard_record['retention']) < 0.9)

    if bool(_guard_record['use_primary']) != _guard_expected_use_primary:
        raise RuntimeError('Frame-retention decision is inconsistent')

_guard_topology = {}

for _guard_movie, _guard_group in _guard_frame.groupby('dataset', sort = True):
    _guard_nodes = _guard_group[_guard_group['row_type'].eq('node')]
    _guard_edges = _guard_group[_guard_group['row_type'].eq('edge')]

    if _guard_nodes.empty or _guard_nodes['t'].lt(0).any():
        raise RuntimeError(f'{_guard_movie}: invalid biological node time')

    if _guard_nodes[['z', 'y', 'x']].lt(0).any().any():
        raise RuntimeError(f'{_guard_movie}: negative biological coordinate')
    _guard_node_time = dict(zip(_guard_nodes['node_id'].astype(int), _guard_nodes['t'].astype(int)))
    _guard_incoming = Counter()
    _guard_outgoing = Counter()

    for _guard_edge in _guard_edges.itertuples():
        _guard_source = int(_guard_edge.source_id)
        _guard_target = int(_guard_edge.target_id)

        if _guard_source not in _guard_node_time or _guard_target not in _guard_node_time or _guard_node_time[_guard_target] != _guard_node_time[_guard_source] + 1:
            raise RuntimeError(f'{_guard_movie}: invalid lineage edge')
        _guard_incoming[_guard_target] += 1
        _guard_outgoing[_guard_source] += 1
    _guard_max_in = max(_guard_incoming.values(), default = 0)
    _guard_max_out = max(_guard_outgoing.values(), default = 0)

    if _guard_max_in > 1 or _guard_max_out > 2:
        raise RuntimeError(f'{_guard_movie}: invalid lineage degree')
    _guard_topology[_guard_movie] = {'nodes': int(len(_guard_nodes)), 'edges': int(len(_guard_edges)), 'max_indegree': int(_guard_max_in), 'max_outdegree': int(_guard_max_out), 'division_parents': int(sum((value == 2 for value in _guard_outgoing.values())))}

_guard_by_movie = {}

for _guard_movie in _guard_expected:
    _guard_movie_records = [row for row in _guard_records if row['dataset'] == _guard_movie]
    _guard_by_movie[_guard_movie] = {'frames': int(len(_guard_movie_records)), 'fallback_frames': int(sum((bool(row['use_primary']) for row in _guard_movie_records))), 'minimum_retention': float(min((row['retention'] for row in _guard_movie_records))), 'median_retention': float(pd.Series([row['retention'] for row in _guard_movie_records]).median())}

_guard_digest = hashlib.sha256(_guard_submission.read_bytes()).hexdigest()
_guard_report = {'experiment': 'secondary_deepcenter_tta_0947_v1', 'status': 'verified_public_lb_0947', 'parent_experiment': 'edge_feature_tta_0946_v1', 'method_attribution': '0.933 fixed-90 dual-seed baseline -> 0.934 harmonic mutual-support fusion -> 0.939 wider divisions and calmer fusion -> 0.941 repair-threshold adaptation -> 0.946 primary edge-feature TTA + held-out post-process selection -> 0.947 secondary feature TTA + DeepCenter TTA', 'source_kernel': 'raykkretzschmar/biohub-bidirectional-primary-union13-diagnostic-v1', 'source_notebook_sha256': '3e65ca691941949196bf417030ea84fccafe16baaec174b2a63540451bb937e8', 'public_output_used': False, 'metric_hack_used': False, 'organizer_labels_used_for_configuration': False, 'leaderboard_feedback_used_for_configuration': True, 'configuration': {'minimum_candidate_retention': 0.9, 'fallback_scope': 'individual_frame', 'detector_threshold': DET_THRESHOLD, 'secondary_detection_weight': float(os.environ.get('BIOHUB_SECONDARY_DETECTION_WEIGHT', '0.80')), 'secondary_edge_weight': float(os.environ.get('BIOHUB_SECONDARY_EDGE_WEIGHT', '0.15')), 'bidirectional_primary_weight': float(os.environ.get('BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT', '0.15')), 'secondary_link_mode': 'low_margin_consensus', 'secondary_low_margin_max': 0.35, 'edge_candidate_threshold': 0.48, 'ilp_appearance_weight': ILP_APPEARANCE_WEIGHT, 'ilp_disappearance_weight': ILP_DISAPPEARANCE_WEIGHT, 'gap_close_um': GAP_CLOSE_UM, 'safe_div_max_um': SAFE_DIV_MAX_UM, 'safe_div_sister_max_um': SAFE_DIV_SISTER_MAX_UM, 'safe_div_sister_symmetry_tau': SAFE_DIV_SISTER_SYMMETRY_TAU, 'deepcenter_safe_div_threshold': DEEPCENTER_SAFE_DIV_THRESHOLD, 'deepcenter_tta': os.environ.get('BIOHUB_DEEPCENTER_TTA', '0') != '0', 'secondary_edge_feature_tta': os.environ.get('BIOHUB_SECONDARY_EDGE_FEATURE_TTA', '0') != '0', 'secondary_edge_feature_tta_weight': float(os.environ.get('BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT', '0.75')), 'deepcenter_gap_threshold': DEEPCENTER_GAP_THRESHOLD, 'deepcenter_gap_confirm_min_span_um': DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM}, 'hardware': {'visible_gpu_count': int(torch.cuda.device_count())}, 'diagnostics': {'rows': int(len(_guard_records)), 'fallback_frames': int(sum((bool(row['use_primary']) for row in _guard_records))), 'by_movie': _guard_by_movie}, 'submission': {'sha256': _guard_digest, 'rows': int(len(_guard_frame)), 'datasets': _guard_datasets}, 'topology': _guard_topology, 'quality_promotion': {'status': 'verified_public_lb_0947', 'required_condition': 'none', 'validated_receipt_sha256': None}}
Path('/kaggle/working/dual_seed_frame_retention_guard_report.json').write_text(json.dumps(_guard_report, indent = 2, sort_keys = True) + '\n')
print(json.dumps(_guard_report, indent = 2, sort_keys = True))

print('PRODUCTION SUBMISSION PIPELINE: COMPLETE')
print('Submission path:', SUBMISSION_PATH)
print('Verified progression preserved: 0.933 -> 0.934 -> 0.939 -> 0.941 -> 0.946 -> 0.947')

