import json
import ast
from pathlib import Path

source_nb_path = Path("competitions/arc_prize_2026/research_sota/keithtyser_nvfp4/duck-qwen3-8-flash-next-nvfp4-mtp.ipynb")
target_nb_path = Path("competitions/arc_prize_2026/exp4_keith_budgetfill_sota/main.ipynb")

with open(source_nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded Keith Tyser source notebook with {len(nb['cells'])} cells.")

# Cell 0: Update markdown title
nb['cells'][0]['source'] = [
    "## Autobot ARC3 Exp 4: Pure Keith Tyser SOTA + Dynamic Per-Game Budget Fill\n",
    "\n",
    "Restores pure Keith Tyser 8.62% SOTA benchmark (full 32K context, 8 sequences, 0 AgentFix regressions)\n",
    "and implements dynamic per-game budget allocation across the full 9-hour execution envelope.\n"
]

# Verify Cell 3 has pure Keith configuration:
src3 = ''.join(nb['cells'][3]['source'])
assert 'PUBLIC25_VLLM_PROFILE_NAME = \'kv5-bf16-mtp3-c8-cg32\'' in src3
assert '"TAAF_VLLM_MAX_NUM_SEQS": "8"' in src3
assert 'LOCAL_ANALYZER_CONTEXT_WINDOW' not in src3  # Full 32K context intact!
print("Cell 3 confirmed: Pure 8-seq, full 32K context.")

# Verify Cell 13 has baseline settings:
src13 = ''.join(nb['cells'][13]['source'])
assert 'bm.solver.max_runtime_s_per_game = 7920.0' in src13
assert 'bm.solver.analyzer_timeout = 900.0' in src13
assert 'bm.solver.concurrency = 28' in src13
print("Cell 13 confirmed: Baseline solver settings (900s analyzer timeout, 28 concurrency).")

# Update Cell 15: replace the competition execution block with Dynamic Budget Fill + Fast Interactive Commit
src15 = ''.join(nb['cells'][15]['source'])

target_split = "if TRUE_SUBMISSION:"
assert target_split in src15, "Target split marker not found in Cell 15"
head_15 = src15[:src15.find(target_split)]

new_tail_15 = """if TRUE_SUBMISSION:
    # Real submission: play the live competition Arcade served by the Kaggle gateway.
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    # The gateway boots asynchronously; wait before swapping in its game list.
    _wait_for_gateway(os.environ["ARC_BASE_URL"])
    bm.games = _competition_games()
    
    # Dynamic per-game budget fill: allocate full 9-hour envelope across waves
    import math
    concurrency = bm.solver.concurrency  # 28
    num_games = len(bm.games)
    waves = math.ceil(num_games / concurrency)
    elapsed = time.time() - NOTEBOOK_START_EPOCH
    # 32400s (9h) notebook budget; 600s soft end + 900s teardown margin = 1500s buffer
    remaining_usable = max(0.0, 32400.0 - elapsed - 1500.0)
    dynamic_per_game = remaining_usable / max(1, waves)
    # Clamp between Keith's baseline 7920s and remaining usable time
    allocated_per_game = max(7920.0, dynamic_per_game)
    bm.solver.max_runtime_s_per_game = allocated_per_game
    print(
        f'AUTOBOT BUDGET FILL: games={num_games} concurrency={concurrency} waves={waves} elapsed={elapsed:.1f}s allocated_per_game={allocated_per_game:.1f}s',
        flush=True,
    )
else:
    print('taaf.kaggle: Fast interactive commit detected. Preparing placeholder submission...', flush=True)
    bm.games = []

bm.n_passes = 1
bm.game_weights = None

# Outside a real submission, stop ~10 min before the wall-clock budget for a graceful exit.
budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
if budget <= 600.0:
    raise RuntimeError(f'Notebook budget is too small for the teardown reserve: {budget}.')
soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(
    seconds=budget - 600.0
)

# Start recovery only after setup readiness and all run gates pass.
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
        print("taaf.kaggle: Fast interactive commit. Writing valid submission.parquet placeholder...", flush=True)
        import pandas as pd
        pd.DataFrame(
            [["1_0", "1", True, 1]],
            columns=["row_id", "game_id", "end_of_game", "score"],
        ).to_parquet(WORKING_DIR / "submission.parquet", index=False)
        print("taaf.kaggle: Valid placeholder submission.parquet created successfully for competition rerun.", flush=True)
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
            )
"""

full_src15 = head_15 + new_tail_15
nb['cells'][15]['source'] = [l + '\n' for l in full_src15.splitlines()]

# Validate AST syntax of all code cells
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])
        test_code = code
        if 'await ' in test_code:
            test_code = "async def __check():\n" + '\n'.join(f"    {l}" for l in test_code.splitlines())
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {idx}: {e}")
            raise e

target_nb_path.parent.mkdir(parents=True, exist_ok=True)
with open(target_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated and validated {target_nb_path} ({len(nb['cells'])} cells).")
