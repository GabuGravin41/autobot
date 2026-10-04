import json
import ast
from pathlib import Path

source_nb_path = Path("competitions/arc_prize_2026/research_sota/keithtyser_nvfp4/duck-qwen3-8-flash-next-nvfp4-mtp.ipynb")
target_nb_path = Path("competitions/arc_prize_2026/exp5_balanced_agentfix_sota/main.ipynb")

with open(source_nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded Keith Tyser source notebook with {len(nb['cells'])} cells.")

# Cell 0: Update markdown title
nb['cells'][0]['source'] = [
    "## Autobot ARC3 Exp 5: Balanced AgentFix SOTA\n",
    "\n",
    "Combines FlashNext NVFP4 MTP engine with AgentFix suite, 16K context, 16 concurrency, and bounded per-game timeout.\n"
]

# Edit Cell 3
src3 = ''.join(nb['cells'][3]['source'])
src3 = src3.replace('"TAAF_VLLM_MAX_NUM_SEQS": "8"', '"TAAF_VLLM_MAX_NUM_SEQS": "16"')
# Add other environment variables before the closing brace of PUBLIC25_VLLM_PROFILE_ENV
env_insertion = """,
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "16384",
    "AGENTFIX_IMAGES": "1",
    "AGENTFIX_MEMORY": "1",
    "AGENTFIX_NOIMPACT": "1",
    "AGENTFIX_MAX_MINUTES": "25"
}"""
src3 = src3.replace('\n}', env_insertion)
nb['cells'][3]['source'] = [l + '\n' for l in src3.splitlines()]

# Edit Cell 13
src13 = ''.join(nb['cells'][13]['source'])
src13 = src13.replace('bm.solver.analyzer_timeout = 900.0', 'bm.solver.analyzer_timeout = 1500.0')
nb['cells'][13]['source'] = [l + '\n' for l in src13.splitlines()]

# Edit Cell 15
src15 = ''.join(nb['cells'][15]['source'])
target_split = "if TRUE_SUBMISSION:"
head_15 = src15[:src15.find(target_split)]

new_tail_15 = """if TRUE_SUBMISSION:
    # Real submission: play the live competition Arcade served by the Kaggle gateway.
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    # The gateway boots asynchronously; wait before swapping in its game list.
    _wait_for_gateway(os.environ["ARC_BASE_URL"])
    bm.games = _competition_games()
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
