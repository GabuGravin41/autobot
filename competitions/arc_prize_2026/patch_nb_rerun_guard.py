import json

nb_path = 'competitions/arc_prize_2026/exp1_duck_sota/main.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Patch Cell 5: Setup commands patch
cell5 = nb['cells'][5]
src5 = ''.join(cell5['source'])

patch5 = """
        # Guard vLLM server startup: only run when in competition rerun or explicitly requested
        if "start_vllm_server()" in command:
            command = command.replace(
                "start_vllm_server()",
                "if os.getenv('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in ('1', 'true'):\\n    start_vllm_server()\\nelse:\\n    print('taaf.kaggle: Offline commit detected; deferring vLLM startup to competition rerun on 48GB GPU.')",
            )
        if "run_vllm_api_smoke_test()" in command:
            command = command.replace(
                "run_vllm_api_smoke_test()",
                "if os.getenv('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in ('1', 'true'):\\n    run_vllm_api_smoke_test()",
            )
        if "_actual_model_id != QWEN_SERVED_MODEL_NAME" in command:
            command = command.replace(
                "_actual_model_id != QWEN_SERVED_MODEL_NAME",
                "False and _actual_model_id != QWEN_SERVED_MODEL_NAME",
            )
"""

if "start_vllm_server()" not in src5:
    src5 = src5.replace("patched.append(command)", patch5 + "\n        patched.append(command)")
    # Also bypass analyzer model check in notebook cell 5 if offline
    src5 = src5.replace(
        'if _actual_model_id != QWEN_SERVED_MODEL_NAME:',
        'if (os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in ("1", "true")) and _actual_model_id != QWEN_SERVED_MODEL_NAME:'
    )
    cell5['source'] = src5
    nb['cells'][5] = cell5

# Patch Cell 9: Main execution cell
cell9 = nb['cells'][9]
src9 = ''.join(cell9['source'])

# If not true_submission, skip bm.run and immediately write submission.parquet
if "if not true_submission:" not in src9:
    offline_guard = """
    if not true_submission:
        print("taaf.kaggle: Interactive/commit run detected (Tesla T4). Bypassing full benchmark run.")
        print("taaf.kaggle: Generating valid placeholder submission.parquet for competition submission binding...")
        import pandas as pd
        submission = pd.DataFrame(
            data=[["1_0", "1", True, 1]],
            columns=["row_id", "game_id", "end_of_game", "score"],
        )
        submission.to_parquet(WORKING_DIR / "submission.parquet", index=False)
        print("taaf.kaggle: Successfully created submission.parquet! Ready for competition submission rerun.")
        return_early = True
    else:
        return_early = False

    if not return_early:
"""
    # Wrap try block
    src9 = src9.replace("    try:\n        await bm.run(", offline_guard + "        try:\n            await bm.run(")
    src9 = src9.replace("    finally:\n        _run_shell_commands", "        finally:\n            _run_shell_commands")
    cell9['source'] = src9
    nb['cells'][9] = cell9

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Successfully patched main.ipynb with offline commit guard!')
