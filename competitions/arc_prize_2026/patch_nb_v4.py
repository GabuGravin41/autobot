import json

nb_path = 'competitions/arc_prize_2026/exp1_duck_sota/main.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][5]
source = ''.join(cell['source'])

source = source.replace('VLLM_MAX_MODEL_LEN = 32768', 'VLLM_MAX_MODEL_LEN = 8192')

patch_args = """
        if "'--max-model-len'," in command:
            command = command.replace(
                "'--max-model-len',",
                "'--enforce-eager',\\n        '--gpu-memory-utilization',\\n        '0.96',\\n        '--max-model-len',",
            )
        if "ANALYZER_CONTEXT_WINDOW = 32768" in command:
            command = command.replace(
                "ANALYZER_CONTEXT_WINDOW = 32768",
                "ANALYZER_CONTEXT_WINDOW = 8192",
            )
"""

target = "        patched.append(command)"
assert target in source, "Target not found in cell 5!"
source = source.replace(target, patch_args + "\n" + target)

cell['source'] = source
nb['cells'][5] = cell

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Successfully patched main.ipynb for Tesla T4 VRAM efficiency!')
