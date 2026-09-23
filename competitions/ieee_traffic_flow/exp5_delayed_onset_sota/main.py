# ==============================================================================
# Autobot Exp 4: Spatio-Temporal Shockwave Queue Model + Spatial-Kalman Hybrid SOTA
# Competition: 2026 IEEE Big Data - Traffic Flow Bench
# ==============================================================================
from __future__ import annotations
import os, sys, gc, time, json, glob, math, random, warnings, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

print('=== AUTOBOT TRAFFIC EXP 4: SPATIO-TEMPORAL SHOCKWAVE PHYSICS SOTA ===')
print(f'Python: {sys.version.split()[0]}')
t0 = time.time()

# ------------------------------------------------------------------------------
# 1. Dataset Root Resolution
# ------------------------------------------------------------------------------
def locate_release_root() -> Path:
    candidates = [
        '/kaggle/input/2026-ieee-big-data-traffic-flow-bench/kaggle_public',
        '/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public',
        '/kaggle/input/2026-ieee-big-data-traffic-flow-bench',
        'kaggle_public', '.'
    ]
    candidates += glob.glob('/kaggle/input/**/kaggle_public', recursive=True)
    candidates += glob.glob('/kaggle/input/**/corridors', recursive=True)
    for c in candidates:
        p = Path(c)
        if (p / 'config' / 'corridors.json').exists() or (p / 'corridors').exists():
            return p
        if p.name == 'corridors' and p.parent.exists():
            return p.parent
    for p in Path('/kaggle/input').rglob('submission_key.csv'):
        return p.parent
    raise FileNotFoundError('Failed to locate competition dataset directory.')

RELEASE_ROOT = locate_release_root()
print(f'Dataset release root: {RELEASE_ROOT}')
SPLIT = 'validation'

manifest_path = RELEASE_ROOT / 'config' / 'corridors.json'
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    PANELS = [p['corridor_id'] for p in manifest['panels']]
else:
    PANELS = sorted([p.name for p in (RELEASE_ROOT / 'corridors').iterdir() if p.is_dir()])
print(f'Corridors ({len(PANELS)}): {PANELS}')

OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
STATE_SUB_PATH = OUTPUT_DIR / 'state_submission.csv'
QUEUE_SUB_PATH = OUTPUT_DIR / 'queue_submission.csv'
ODME_SUB_PATH = OUTPUT_DIR / 'odme_submission.csv'
FINAL_SUB_PATH = OUTPUT_DIR / 'submission.csv'

# ------------------------------------------------------------------------------
# 2. Topology & Corridor Structure Helpers
# ------------------------------------------------------------------------------
def get_corridor_link_order(panel_dir: Path) -> list[str]:
    links_path = panel_dir / 'network' / 'links.csv'
    if links_path.exists():
        ldf = pd.read_csv(links_path, dtype={'link_id': str})
        if 'milepost' in ldf.columns:
            ldf['milepost'] = pd.to_numeric(ldf['milepost'], errors='coerce')
            return ldf.sort_values(['milepost', 'link_id'])['link_id'].tolist()
        return ldf['link_id'].drop_duplicates().tolist()
    return []

def get_upstream_map(panel_dir: Path) -> dict[str, list[str]]:
    topo_path = panel_dir / 'network' / 'lwr_mainline_topology.csv'
    upstream = {}
    if topo_path.exists():
        topo = pd.read_csv(topo_path, dtype=str)
        for row in topo.itertuples(index=False):
            lid = str(row.link_id)
            in_ids = [x.strip() for x in str(getattr(row, 'incoming_link_ids', '')).split(';') if x.strip()]
            upstream[lid] = in_ids
    return upstream

def thresholds(panel_dir: Path, links: pd.DataFrame) -> dict[str, float]:
    fallback = pd.to_numeric(links['free_speed_kmh'], errors='coerce').fillna(105.0) * 0.60
    values = dict(zip(links['link_id'].astype(str), fallback.astype(float)))
    fd = panel_dir / 'network' / 'fd_parameters.csv'
    if fd.exists():
        frame = pd.read_csv(fd)
        if 'v_cut' in frame.columns:
            for row in frame.dropna(subset=['link_id', 'v_cut']).itertuples(index=False):
                values[str(row.link_id)] = float(row.v_cut)
    return values

# ------------------------------------------------------------------------------
# 3. Kalman & Profile Utilities
# ------------------------------------------------------------------------------
def files(panel_dir: Path, split: str) -> list[Path]:
    direct = sorted((panel_dir / split / 'mainline_states').glob('**/*.parquet'))
    if direct:
        return direct
    return sorted((panel_dir / split / 'mainline_states_masked').glob('**/*.parquet'))

def masked_files(panel_dir: Path, split: str) -> list[Path]:
    return sorted((panel_dir / split / 'mainline_states_masked').glob('**/*.parquet'))

def slot_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(frame['timestamp'], utc=True)
    weekday = ts.dt.weekday.to_numpy(dtype=np.int16)
    tod = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(dtype=np.int16)
    return weekday, tod

def kalman_smooth(y: np.ndarray, q: float, r: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    finite = np.flatnonzero(np.isfinite(y))
    if len(finite) == 0:
        return np.zeros(n)
    x_f = np.zeros(n)
    p_f = np.zeros(n)
    x_p = np.zeros(n)
    p_p = np.zeros(n)
    x, p = y[finite[0]], r
    for t in range(n):
        if t > 0:
            p = p + q
        x_p[t], p_p[t] = x, p
        if np.isfinite(y[t]):
            k = p / (p + r)
            x = x + k * (y[t] - x)
            p = (1.0 - k) * p
        x_f[t], p_f[t] = x, p
    x_s = x_f.copy()
    for t in range(n - 2, -1, -1):
        g = p_f[t] / p_p[t + 1]
        x_s[t] = x_f[t] + g * (x_s[t + 1] - x_f[t + 1])
    return x_s

def smooth_deviation_rows(dev: np.ndarray) -> np.ndarray:
    out = np.zeros_like(dev)
    for i in range(dev.shape[0]):
        row = dev[i]
        if not np.isfinite(row).any():
            continue
        sd = max(float(np.nanstd(row)), 1e-3)
        out[i] = kalman_smooth(row, q=(0.35 * sd) ** 2, r=(0.8 * sd) ** 2)
    return out

def build_profile(panel: str, panel_dir: Path) -> tuple[dict, dict, dict]:
    train_files = files(panel_dir, 'train')
    first = pd.read_parquet(train_files[0], columns=['link_id'])
    link_ids = sorted(first.link_id.astype(str).unique())
    link_index = {link_id: i for i, link_id in enumerate(link_ids)}
    n_links = len(link_ids)
    n_slots = 7 * 288
    speed_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    speed_count = np.zeros((n_links, n_slots), dtype=np.int64)
    flow_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    flow_count = np.zeros((n_links, n_slots), dtype=np.int64)
    for path in train_files:
        frame = pd.read_parquet(
            path,
            columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'pct_observed'],
        )
        frame['link_id'] = frame['link_id'].astype(str)
        li = frame.link_id.map(link_index).to_numpy(dtype=np.int64)
        weekday, tod = slot_values(frame)
        slot = weekday * 288 + tod
        valid = (
            pd.to_numeric(frame.pct_observed, errors='coerce').ge(75)
            & frame.speed_kmh.notna()
            & frame.flow_vph.notna()
        ).to_numpy()
        sp = pd.to_numeric(frame.speed_kmh, errors='coerce').to_numpy(dtype=float)
        fl = pd.to_numeric(frame.flow_vph, errors='coerce').to_numpy(dtype=float)
        valid_sp = valid & np.isfinite(sp)
        valid_fl = valid & np.isfinite(fl)
        np.add.at(speed_sum, (li[valid_sp], slot[valid_sp]), sp[valid_sp])
        np.add.at(speed_count, (li[valid_sp], slot[valid_sp]), 1)
        np.add.at(flow_sum, (li[valid_fl], slot[valid_fl]), fl[valid_fl])
        np.add.at(flow_count, (li[valid_fl], slot[valid_fl]), 1)
    speed_mean = np.zeros_like(speed_sum)
    flow_mean = np.zeros_like(flow_sum)
    np.divide(speed_sum, np.maximum(speed_count, 1), out=speed_mean, where=speed_count > 0)
    np.divide(flow_sum, np.maximum(flow_count, 1), out=flow_mean, where=flow_count > 0)
    speed_link = np.divide(speed_sum.sum(axis=1), np.maximum(speed_count.sum(axis=1), 1))
    flow_link = np.divide(flow_sum.sum(axis=1), np.maximum(flow_count.sum(axis=1), 1))
    return (
        {'link_ids': link_ids, 'link_index': link_index, 'mean': speed_mean, 'fallback': speed_link},
        {'link_ids': link_ids, 'link_index': link_index, 'mean': flow_mean, 'fallback': flow_link},
        {'speed_count': speed_count, 'flow_count': flow_count},
    )

# ------------------------------------------------------------------------------
# 4. Task 1: Spatial-Kalman Hybrid State Reconstruction
# ------------------------------------------------------------------------------
print('\n>>> [TASK 1] Building Spatial-Kalman Hybrid State Predictions...')
STATE_COLUMNS = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']
if STATE_SUB_PATH.exists():
    STATE_SUB_PATH.unlink()

total_state_rows = 0
first_write = True

for panel_idx, panel in enumerate(PANELS, 1):
    panel_dir = RELEASE_ROOT / 'corridors' / panel
    print(f'[{panel_idx}/{len(PANELS)}] Processing Corridor {panel}...')
    speed_profile, flow_profile, profile_counts = build_profile(panel, panel_dir)
    link_ids = speed_profile['link_ids']
    link_index = speed_profile['link_index']
    n_links = len(link_ids)

    # Ordered links along corridor for spatial interpolation
    corridor_order = get_corridor_link_order(panel_dir)
    order_map = {lid: i for i, lid in enumerate(corridor_order) if lid in link_index}

    template_path = RELEASE_ROOT / 'task1' / panel / SPLIT / 'sample_submission_state.csv'
    template = pd.read_csv(template_path, usecols=['timestamp', 'station_id', 'link_id'], dtype=str)
    template_keys = set(zip(template.timestamp, template.station_id, template.link_id))

    panel_written = 0
    for path in masked_files(panel_dir, SPLIT):
        frame = pd.read_parquet(
            path,
            columns=[
                'date', 'timestamp', 'station_id', 'link_id', 'speed_kmh',
                'flow_vph', 'is_score_eligible', 'mask_regime',
            ],
        )
        frame['station_id'] = frame.station_id.astype(str)
        frame['link_id'] = frame.link_id.astype(str)
        weekday, tod = slot_values(frame)
        slot = weekday * 288 + tod
        li = frame.link_id.map(link_index).fillna(-1).to_numpy(dtype=np.int64)
        known = li >= 0
        safe_li = np.where(known, li, 0)

        base_speed = speed_profile['mean'][safe_li, slot]
        base_flow = flow_profile['mean'][safe_li, slot]
        base_speed = np.where(profile_counts['speed_count'][safe_li, slot] == 0, speed_profile['fallback'][safe_li], base_speed)
        base_flow = np.where(profile_counts['flow_count'][safe_li, slot] == 0, flow_profile['fallback'][safe_li], base_flow)

        raw_speed = pd.to_numeric(frame.speed_kmh, errors='coerce').to_numpy(dtype=float)
        raw_flow = pd.to_numeric(frame.flow_vph, errors='coerce').to_numpy(dtype=float)
        eligible = frame.is_score_eligible.astype(bool).to_numpy()

        blanked = np.fromiter(
            (k in template_keys for k in zip(frame.timestamp.astype(str), frame.station_id.astype(str), frame.link_id.astype(str))),
            dtype=bool, count=len(frame)
        )

        for regime in pd.unique(frame.mask_regime.astype(str)):
            target = blanked & (frame.mask_regime.astype(str) == regime).to_numpy()
            
            # Temporal Kalman deviations
            dev_speed = np.full((n_links, 288), np.nan)
            dev_flow = np.full((n_links, 288), np.nan)
            dev_speed_n = np.zeros((n_links, 288), dtype=np.int32)
            dev_flow_n = np.zeros((n_links, 288), dtype=np.int32)
            dev_speed_sum = np.zeros((n_links, 288), dtype=float)
            dev_flow_sum = np.zeros((n_links, 288), dtype=float)

            obs_speed = eligible & ~target & known & np.isfinite(raw_speed) & np.isfinite(base_speed)
            obs_flow = eligible & ~target & known & np.isfinite(raw_flow) & np.isfinite(base_flow)

            np.add.at(dev_speed_sum, (li[obs_speed], tod[obs_speed]), (raw_speed - base_speed)[obs_speed])
            np.add.at(dev_flow_sum, (li[obs_flow], tod[obs_flow]), (raw_flow - base_flow)[obs_flow])
            np.add.at(dev_speed_n, (li[obs_speed], tod[obs_speed]), 1)
            np.add.at(dev_flow_n, (li[obs_flow], tod[obs_flow]), 1)

            np.divide(dev_speed_sum, dev_speed_n, out=dev_speed, where=dev_speed_n > 0)
            np.divide(dev_flow_sum, dev_flow_n, out=dev_flow, where=dev_flow_n > 0)

            smooth_speed = smooth_deviation_rows(dev_speed)
            smooth_flow = smooth_deviation_rows(dev_flow)

            kalman_pred_speed = np.clip(base_speed + smooth_speed[safe_li, tod], 10.0, 130.0)
            kalman_pred_flow = np.clip(base_flow + smooth_flow[safe_li, tod], 50.0, None)

            # Spatial neighborhood blending at instantaneous timestamp t
            final_pred_speed = kalman_pred_speed.copy()
            final_pred_flow = kalman_pred_flow.copy()

            # Iterate unique tod slices to apply spatial correlation where neighbors exist
            for cur_tod in np.unique(tod[target]):
                tod_mask = (tod == cur_tod)
                obs_tod_mask = tod_mask & obs_speed
                if obs_tod_mask.sum() >= 3:
                    # Spatial interpolation along link order
                    obs_order = [order_map[lid] for lid in frame.link_id[obs_tod_mask] if lid in order_map]
                    obs_sp = raw_speed[obs_tod_mask]
                    if len(obs_order) >= 3:
                        sort_idx = np.argsort(obs_order)
                        x_pts = np.array(obs_order)[sort_idx]
                        y_pts = obs_sp[sort_idx]
                        
                        target_tod_mask = tod_mask & target
                        for tidx in np.flatnonzero(target_tod_mask):
                            lid = frame.link_id.iloc[tidx]
                            if lid in order_map:
                                pos = order_map[lid]
                                sp_interp = float(np.interp(pos, x_pts, y_pts))
                                # Hybrid blend 60% Kalman profile + 40% spatial observation
                                final_pred_speed[tidx] = 0.60 * kalman_pred_speed[tidx] + 0.40 * sp_interp

            final_pred_speed = np.where(known, final_pred_speed, base_speed)
            final_pred_flow = np.where(known, final_pred_flow, base_flow)
            final_pred_speed = np.clip(final_pred_speed, 10.0, 130.0)
            final_pred_flow = np.clip(final_pred_flow, 50.0, None)

            mask = target & np.isfinite(final_pred_speed) & np.isfinite(final_pred_flow)
            if not mask.any():
                continue

            out = pd.DataFrame(
                {
                    'panel': panel,
                    'timestamp': frame.loc[mask, 'timestamp'].astype(str).to_numpy(),
                    'station_id': frame.loc[mask, 'station_id'].to_numpy(),
                    'link_id': frame.loc[mask, 'link_id'].to_numpy(),
                    'mask_regime': regime,
                    'speed_kmh': final_pred_speed[mask],
                    'flow_vph': final_pred_flow[mask],
                },
                columns=STATE_COLUMNS,
            )
            out.to_csv(STATE_SUB_PATH, mode='w' if first_write else 'a', header=first_write, index=False)
            first_write = False
            panel_written += len(out)
            total_state_rows += len(out)
    print(f'  -> Corridor {panel}: wrote {panel_written:,} state rows.')

print(f'Total Task 1 rows written: {total_state_rows:,}')

# ------------------------------------------------------------------------------
# 5. Task 2: Spatio-Temporal Shockwave Queue Model
# ------------------------------------------------------------------------------
print('\n>>> [TASK 2] Building Spatio-Temporal Shockwave Queue Model...')
def _task2_parts(root: Path, splits: list[str], name: str) -> list[Path]:
    parts = []
    for panel_dir in sorted(p for p in (root / 'task2').glob('*') if p.is_dir()):
        for split in splits:
            candidate = panel_dir / split / name
            if candidate.exists():
                parts.append(candidate)
    return parts

def read_window_index(root: Path, splits: list[str]) -> pd.DataFrame:
    flat = root / 'task2' / 'window_index.csv'
    if flat.exists():
        frame = pd.read_csv(flat)
        return frame[frame.split.isin(splits)].copy()
    parts = _task2_parts(root, splits, 'window_index.csv')
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

def read_window_history(root: Path, splits: list[str]) -> pd.DataFrame:
    flat = root / 'task2' / 'window_history.parquet'
    columns = ['window_id', 'timestamp', 'link_id', 'speed_kmh', 'is_score_eligible']
    if flat.exists():
        return pd.read_parquet(flat, columns=columns)
    parts = _task2_parts(root, splits, 'window_history.parquet')
    return pd.concat([pd.read_parquet(p, columns=columns) for p in parts], ignore_index=True)

def read_queue_template(root: Path, splits: list[str]) -> pd.DataFrame:
    columns = ['window_id', 'timestamp', 'link_id']
    for flat in (root / 'task2' / 'sample_submission_queue.csv',
                 root / 'submission_templates_per_task' / 'sample_submission_queue.csv'):
        if flat.exists():
            return pd.read_csv(flat, usecols=columns)
    parts = _task2_parts(root, splits, 'sample_submission_queue.csv')
    return pd.concat([pd.read_csv(p, usecols=columns) for p in parts], ignore_index=True)

splits = [SPLIT]
windows = read_window_index(RELEASE_ROOT, splits)
template_q = read_queue_template(RELEASE_ROOT, splits)
template_q['window_id'] = template_q.window_id.astype(str)
template_q['link_id'] = template_q.link_id.astype(str)
template_q['timestamp'] = pd.to_datetime(template_q.timestamp, utc=True)

history = read_window_history(RELEASE_ROOT, splits)
history['window_id'] = history.window_id.astype(str)
history['timestamp'] = pd.to_datetime(history.timestamp, utc=True)
history['link_id'] = history.link_id.astype(str)
history['speed_kmh'] = pd.to_numeric(history.speed_kmh, errors='coerce')

panel_by_window = windows.set_index('window_id').panel.astype(str).to_dict()
thresholds_by_panel = {}
upstream_by_panel = {}
corridor_order_by_panel = {}

for panel in sorted(windows.panel.astype(str).unique()):
    panel_dir = RELEASE_ROOT / 'corridors' / panel
    links = pd.read_csv(panel_dir / 'network' / 'links.csv')
    links['link_id'] = links.link_id.astype(str)
    thresholds_by_panel[panel] = thresholds(panel_dir, links)
    upstream_by_panel[panel] = get_upstream_map(panel_dir)
    corridor_order_by_panel[panel] = get_corridor_link_order(panel_dir)

q_targets = []
for window_id, h in history.groupby('window_id', sort=True):
    panel = panel_by_window.get(str(window_id))
    if panel is None:
        continue
    
    thresh = thresholds_by_panel[panel]
    up_map = upstream_by_panel[panel]
    order_list = corridor_order_by_panel[panel]
    
    # Get observation at forecast origin T and 15 mins prior (T-15m)
    timestamps = sorted(h.timestamp.unique())
    last_time = timestamps[-1]
    prev_time = timestamps[-4] if len(timestamps) >= 4 else timestamps[0]

    at_T = h[h.timestamp.eq(last_time)].copy()
    at_T['link_id'] = at_T.link_id.astype(str)
    at_prev = h[h.timestamp.eq(prev_time)].copy()
    at_prev['link_id'] = at_prev.link_id.astype(str)
    
    speed_T = dict(zip(at_T.link_id, at_T.speed_kmh))
    speed_prev = dict(zip(at_prev.link_id, at_prev.speed_kmh))
    
    # Check queue status at origin T
    at_T['queue_now'] = at_T.speed_kmh <= at_T.link_id.map(thresh).fillna(63.0)
    at_T['queue_now'] &= at_T.is_score_eligible.astype(bool)
    queued_at_T = set(at_T.loc[at_T.queue_now, 'link_id'])
    
    target = template_q[template_q.window_id == str(window_id)][['window_id', 'timestamp', 'link_id']].copy()
    future_times = sorted(target.timestamp.unique())

    # Top-2 empirical recurrent bottlenecks identified across all 8 corridors from ground truth training data
    EMPIRICAL_TOP2_BOTTLENECKS = {
        'D12_I5_N': ['L5N-059', 'L5N-104'],
        'D12_I5_S': ['L5S-152', 'L5S-249'],
        'D7_I10_E': ['L10E-043', 'L10E-125'],
        'D7_I10_W': ['L10W-013', 'L10W-212'],
        'D7_I210_E': ['L210E-132', 'L210E-261'],
        'D7_I210_W': ['L210W-190', 'L210W-246'],
        'D7_I405_N': ['L405N-145', 'L405N-041'],
        'D7_I405_S': ['L405S-264', 'L405S-127'],
    }

    cond_map = windows.set_index('window_id').condition.astype(str).to_dict() if 'condition' in windows.columns else {}
    cond_val = cond_map.get(str(window_id), '').lower()
    is_ongoing = (len(queued_at_T) > 0) or ('ongoing' in cond_val)

    preds_by_time = {}
    if is_ongoing:
        # Mode A: Queue Ongoing
        # Steps 0-2 (T+5m to T+15m): Pure persistence of active queue links
        # Steps 3-5 (T+20m to T+30m): Backward shockwave expansion by 1 link upstream along corridor
        active_queue = set(queued_at_T)
        for step_idx, ftime in enumerate(future_times):
            new_active = set(active_queue)
            if step_idx >= 3:
                for qlid in list(active_queue):
                    if qlid in order_list:
                        idx = order_list.index(qlid)
                        if idx > 0:
                            prev_node = order_list[idx - 1]
                            sp_p = speed_T.get(prev_node, 100.0)
                            th_p = thresh.get(prev_node, 63.0)
                            if sp_p <= th_p * 1.08:
                                new_active.add(prev_node)
            active_queue = new_active
            preds_by_time[ftime] = dict.fromkeys(active_queue, 1)

    else:
        # Mode B: Queue Onset
        # Physics Rule: Vehicle accumulation to breakdown requires 25-30 mins delay.
        # Steps 0-3 (T+5m to T+20m): Zero queue (100% precision, eliminating false alarm IoU=0 collapse)
        # Steps 4-5 (T+25m, T+30m): Activate breakdown on primary recurrent bottlenecks
        bottlenecks = list(EMPIRICAL_TOP2_BOTTLENECKS.get(panel, []))
        if not bottlenecks:
            # Fallback dynamic bottleneck discovery
            ratios = {lid: speed_T[lid] / thresh.get(lid, 63.0) for lid in order_list if lid in speed_T and thresh.get(lid, 63.0) > 0}
            bottlenecks = sorted(ratios.keys(), key=lambda x: ratios[x])[:2]

        for step_idx, ftime in enumerate(future_times):
            dt_min = (ftime - last_time).total_seconds() / 60.0
            if dt_min >= 24.5 and bottlenecks:
                # Commences breakdown on primary bottleneck
                active_set = {bottlenecks[0]}
                if dt_min >= 29.5 and len(bottlenecks) > 1:
                    active_set.add(bottlenecks[1])
                preds_by_time[ftime] = dict.fromkeys(active_set, 1)
            else:
                preds_by_time[ftime] = {}

    # Map predictions to template rows
    target['queue_pred'] = 0
    for ftime, qdict in preds_by_time.items():
        tmask = target.timestamp == ftime
        if tmask.any() and qdict:
            target.loc[tmask, 'queue_pred'] = target.loc[tmask, 'link_id'].map(qdict).fillna(0).astype(int)

    q_targets.append(target)

queue_df = pd.concat(q_targets, ignore_index=True).drop_duplicates(['window_id', 'timestamp', 'link_id'])
queue_df.to_csv(QUEUE_SUB_PATH, index=False)
print(f'Wrote Task 2 Spatio-Temporal Queue predictions: {len(queue_df):,} rows.')

# ------------------------------------------------------------------------------
# 6. Task 4: Regularized ODME with Prior & Observed Connector Masking
# ------------------------------------------------------------------------------
print('\n>>> [TASK 4] Solving Regularized ODME Path Flows...')
REG_LAMBDA = 0.05

def load_operator(network: Path):
    paths = pd.read_csv(network / 'path_set.csv')
    paths['path_id'] = paths.path_id.astype(str)
    incidence = pd.read_csv(network / 'path_link_incidence.csv')
    incidence['path_id'] = incidence.path_id.astype(str)
    incidence['link_id'] = incidence.link_id.astype(str)
    path_ids = paths.path_id.tolist()
    link_ids = incidence.link_id.drop_duplicates().tolist()
    path_index = {x: i for i, x in enumerate(path_ids)}
    link_index = {x: i for i, x in enumerate(link_ids)}
    rr = incidence.link_id.map(link_index).to_numpy(dtype=np.int64)
    cc = incidence.path_id.map(path_index).to_numpy(dtype=np.int64)
    A = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
    A[rr, cc] = 1.0
    return path_ids, link_ids, A, paths

def released_counts(release: Path, panel: str, split: str) -> pd.DataFrame | None:
    path = release / 'task4' / panel / split / 'synthetic_link_counts.csv'
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={'link_id': str})
    return frame[['link_id', 'count']]

def released_prior(release: Path, panel: str, split: str) -> pd.DataFrame | None:
    path = release / 'task4' / panel / split / 'synthetic_weak_prior.csv'
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={'path_id': str})

odme_parts = []
for panel in PANELS:
    panel_dir = RELEASE_ROOT / 'corridors' / panel
    network = panel_dir / 'network'
    path_ids, link_ids, A, paths = load_operator(network)
    
    prior_df = released_prior(RELEASE_ROOT, panel, SPLIT)
    if prior_df is not None:
        prior_map = prior_df.set_index(prior_df.path_id.astype(str))['path_flow'].to_dict()
        base_values = np.array([float(prior_map.get(pid, 0.0)) for pid in path_ids], dtype=float)
        departure_time = str(prior_df.departure_time.iloc[0])
    else:
        base_values = np.zeros(len(path_ids), dtype=float)
        departure_time = 'PUBLIC-VAL-PM'

    val_counts_df = released_counts(RELEASE_ROOT, panel, SPLIT)
    if val_counts_df is not None:
        val_counts_map = val_counts_df.set_index('link_id')['count'].to_dict()
        val_counts = np.array([float(val_counts_map.get(lid, 0.0)) for lid in link_ids], dtype=float)
        observed_links = set(val_counts_df.link_id)
    else:
        val_counts = np.zeros(len(link_ids), dtype=float)
        observed_links = set()

    measured = np.array([link in observed_links for link in link_ids], dtype=bool)
    if not measured.any():
        measured = np.ones(len(link_ids), dtype=bool)

    aa = np.vstack([A[measured], np.sqrt(REG_LAMBDA) * np.eye(A.shape[1])])
    bb = np.concatenate([val_counts[measured], np.sqrt(REG_LAMBDA) * base_values])
    val_f, _ = nnls(aa, bb)

    od_frame = paths[['path_id', 'origin_zone', 'destination_zone']].copy()
    od_frame['panel'] = panel
    od_frame['departure_time'] = departure_time
    od_frame['path_flow'] = val_f
    odme_parts.append(od_frame[['panel', 'departure_time', 'path_id', 'origin_zone', 'destination_zone', 'path_flow']])

odme_df = pd.concat(odme_parts, ignore_index=True)
odme_df.to_csv(ODME_SUB_PATH, index=False)
print(f'Wrote Task 4 ODME predictions: {len(odme_df):,} rows.')

# ------------------------------------------------------------------------------
# 7. Merge All Tasks via Streaming Submission Key
# ------------------------------------------------------------------------------
print('\n>>> [MERGE] Streaming submission_key.csv to build final submission.csv...')
CHUNK = 1_000_000
OUT_COLUMNS = ['submission_id', 'task', 'speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']

KEYS = {
    'state': ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'],
    'queue': ['window_id', 'timestamp', 'link_id'],
    'odme': ['panel', 'departure_time', 'path_id']
}
VALUES = {'state': ['speed_kmh', 'flow_vph'], 'queue': ['queue_pred'], 'odme': ['path_flow']}

def load_table(path: Path, task: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in KEYS[task]:
        df[col] = pd.to_datetime(df[col], utc=True) if col == 'timestamp' else df[col].astype(str)
    for col in VALUES[task]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.set_index(KEYS[task])

tasks_map = {
    'state': load_table(STATE_SUB_PATH, 'state'),
    'queue': load_table(QUEUE_SUB_PATH, 'queue'),
    'odme': load_table(ODME_SUB_PATH, 'odme')
}

key_path = RELEASE_ROOT / 'submission_key.csv'
if not key_path.exists():
    for p in Path('/kaggle/input').rglob('submission_key.csv'):
        key_path = p
        break

print(f'Using submission key from: {key_path}')
first_chunk = True
written = 0
gaps = {t: 0 for t in tasks_map}

for chunk in pd.read_csv(key_path, chunksize=CHUNK, dtype=str):
    chunk['timestamp'] = pd.to_datetime(chunk.timestamp, utc=True, errors='coerce')
    out = pd.DataFrame({
        'submission_id': chunk.submission_id.astype('int64'),
        'task': chunk.task.astype(str)
    })
    for col in ('speed_kmh', 'flow_vph', 'queue_pred', 'path_flow'):
        out[col] = 0.0
    for task, tbl in tasks_map.items():
        rows = (out.task == task).to_numpy()
        if not rows.any():
            continue
        wanted = pd.MultiIndex.from_frame(chunk.loc[rows, KEYS[task]])
        found = tbl.reindex(wanted)
        for col in VALUES[task]:
            vals = found[col].to_numpy()
            gaps[task] += int(pd.isna(vals).sum())
            out.loc[rows, col] = pd.Series(vals).fillna(0.0).to_numpy()

    out[OUT_COLUMNS].to_csv(FINAL_SUB_PATH, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
    first_chunk = False
    written += len(out)
    print(f'  Processed and wrote {written:,} rows...', flush=True)

print(f'\nSuccessfully generated {FINAL_SUB_PATH}: {written:,} rows.')
for task, g in gaps.items():
    if g > 0:
        print(f'WARNING: {g:,} {task} rows were missing in lookup!')
    else:
        print(f'PASS: {task} has 0 gaps.')

sub_size_mb = FINAL_SUB_PATH.stat().st_size / (1024 * 1024)
print(f'File size: {sub_size_mb:.2f} MB')
print(f'Total execution time: {time.time() - t0:.1f}s')
print('=== PIPELINE COMPLETE ===')
