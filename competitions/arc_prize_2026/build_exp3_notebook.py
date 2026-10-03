import json
import ast
from pathlib import Path

source_nb_path = Path("competitions/arc_prize_2026/research_sota/scottlegrand/taaf-flashnext-sheetu12b-0922.ipynb")
target_nb_path = Path("competitions/arc_prize_2026/exp3_flashnext_agentfix_sota/main.ipynb")

with open(source_nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded source notebook with {len(nb['cells'])} cells.")

# Modify cell 14: increase analyzer_timeout to 1200.0s for deeper puzzle exploration
src14 = ''.join(nb['cells'][14]['source'])
src14 = src14.replace("bm.solver.analyzer_timeout = 900.0", "bm.solver.analyzer_timeout = 1200.0")
nb['cells'][14]['source'] = [l + '\n' for l in src14.splitlines()]

# Modify cell 16: offline commit bypass
src16 = ''.join(nb['cells'][16]['source'])

# 1. Update gateway / game setup for offline branch
old_else_block = """else:
    # Interactive run: play the bundled competition environments offline (no gateway).
    # The competition's environment files ship alongside the wheelhouse in the competition dataset.
    competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
    offline_games = _offline_games(competition_env_files)
    offline_by_id = {game.env_name: game for game in offline_games}
    if len(offline_by_id) != len(offline_games):
        raise RuntimeError('The offline public game list contains duplicate IDs.')
    missing = sorted(set(PUBLIC_GAME_IDS) - set(offline_by_id))
    extra = sorted(set(offline_by_id) - set(PUBLIC_GAME_IDS))
    if missing or extra:
        raise RuntimeError(
            f'Offline public game set changed; missing={missing}, extra={extra}.'
        )
    bm.games = [offline_by_id[game_id] for game_id in PUBLIC_GAME_IDS]
    if len(bm.games) != 25:
        raise RuntimeError(f'Expected 25 public games, got {len(bm.games)}.')
    print(f'PUBLIC25_SELECTION games={len(bm.games)} passes=1', flush=True)"""

new_else_block = """else:
    print('taaf.kaggle: Offline interactive commit detected. Preparing placeholder submission...', flush=True)
    bm.games = []"""

assert old_else_block in src16, "old_else_block not found in src16"
src16 = src16.replace(old_else_block, new_else_block)

# 2. Replace the execution and audit block cleanly
old_tail = """# Start recovery only after setup readiness and all run gates pass.
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))
import vllm_server_watchdog as vllm_watchdog

vllm_watchdog_setup = vllm_watchdog.load_setup(BUNDLE_DIR / 'serving_setup.py')
vllm_watchdog.start_background(
    vllm_watchdog_setup,
    vllm_watchdog.WatchdogConfig(
        interval_seconds=15.0,
        request_timeout_seconds=5,
        failure_threshold=4,
        max_restart_attempts=2,
    ),
)

# Play the benchmark; watchdog stop and teardown run even if it raises.
try:
    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)
    bm._save_json()
    if not TRUE_SUBMISSION:
        # Kaggle Save & Run expects this valid placeholder after an offline run.
        # A real competition rerun uses the live gateway and never enters this branch.
        import pandas as pd

        pd.DataFrame(
            [["1_0", "1", True, 1]],
            columns=["row_id", "game_id", "end_of_game", "score"],
        ).to_parquet(WORKING_DIR / "submission.parquet", index=False)

        # Check terminal coverage, then call the frozen scorer once.
        # This path does not render HTML.
        public_runs = list(bm.game_runs)
        public_run_ids = [run.game_id for run in public_runs]
        if len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS):
            raise RuntimeError(
                f'Public run coverage changed: count={len(public_runs)} ids={public_run_ids}.'
            )
        unfinished = [
            (run.game_id, run.state, run.final_score)
            for run in public_runs
            if run.state not in {'won', 'gave_up', 'cancelled'}
            or run.final_score is None
        ]
        if unfinished:
            raise RuntimeError(f'Public runs did not finalize cleanly: {unfinished}.')
        crashed = [run.game_id for run in public_runs if run.state == 'crashed']
        if crashed:
            raise RuntimeError(f'Public runs crashed: {crashed}.')
        total_actions = sum(len(run.history) for run in public_runs)
        if total_actions <= 0:
            raise RuntimeError('Public runs produced no actions.')

        from inference.tools.eval import evaluate_runs, save_score_file

        score_summary = evaluate_runs([WORKING_DIR])
        score_path = save_score_file(
            score_summary,
            run_dirs=[WORKING_DIR],
            output_path=WORKING_DIR / "score.json",
        )
        if Path(score_path) != WORKING_DIR / 'score.json' or not Path(score_path).is_file():
            raise RuntimeError(f'Frozen scorer did not write score.json: {score_path}.')
        print(
            f'PUBLIC25_AUDIT runs=25 actions={total_actions} score_path={score_path}',
            flush=True,
        )
finally:
    try:
        vllm_watchdog.stop_background(timeout_seconds=15.0)
    finally:
        for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
            print(f"taaf.kaggle: teardown command: {command}", flush=True)
            subprocess.run(
                command,
                shell=True,
                check=False,
                cwd=WORKING_DIR,
                env=_command_env(),
                timeout=30.0,
            )"""

new_tail = """# Start recovery only after setup readiness and all run gates pass.
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))
import vllm_server_watchdog as vllm_watchdog

if TRUE_SUBMISSION:
    vllm_watchdog_setup = vllm_watchdog.load_setup(BUNDLE_DIR / 'serving_setup.py')
    vllm_watchdog.start_background(
        vllm_watchdog_setup,
        vllm_watchdog.WatchdogConfig(
            interval_seconds=15.0,
            request_timeout_seconds=5,
            failure_threshold=4,
            max_restart_attempts=2,
        ),
    )

# Play the benchmark; watchdog stop and teardown run even if it raises.
try:
    if TRUE_SUBMISSION:
        print("taaf.kaggle: LIVE COMPETITION RERUN DETECTED! Playing live arcade gateway...", flush=True)
        await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)
        bm._save_json()
    else:
        print("taaf.kaggle: Offline interactive commit detected. Writing valid submission.parquet placeholder...", flush=True)
        import pandas as pd
        pd.DataFrame(
            [["1_0", "1", True, 1]],
            columns=["row_id", "game_id", "end_of_game", "score"],
        ).to_parquet(WORKING_DIR / "submission.parquet", index=False)
        print("taaf.kaggle: Placeholder submission.parquet created successfully for competition rerun.", flush=True)
finally:
    try:
        if TRUE_SUBMISSION:
            vllm_watchdog.stop_background(timeout_seconds=15.0)
    finally:
        for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
            print(f"taaf.kaggle: teardown command: {command}", flush=True)
            subprocess.run(
                command,
                shell=True,
                check=False,
                cwd=WORKING_DIR,
                env=_command_env(),
                timeout=30.0,
            )"""

assert old_tail in src16, "old_tail not found in src16"
src16 = src16.replace(old_tail, new_tail)

nb['cells'][16]['source'] = [l + '\n' for l in src16.splitlines()]

# Validate entire notebook code syntax
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])
        # Replace top-level 'await' with dummy async def for syntax check
        test_code = code
        if 'await ' in test_code:
            test_code = f"async def __check():\n" + '\n'.join(f"    {l}" for l in test_code.splitlines())
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {idx}: {e}")
            raise e

target_nb_path.parent.mkdir(parents=True, exist_ok=True)
with open(target_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated and validated {target_nb_path} with {len(nb['cells'])} cells.")
