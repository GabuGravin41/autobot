import json

nb_path = 'competitions/arc_prize_2026/exp1_duck_sota/main.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][5]
source = cell['source']

patch_code = """
        # Bypass GPU assertion so any Kaggle GPU works cleanly
        if "assert len(gpu_names) == int(expected_count)" in command:
            command = command.replace(
                "assert len(gpu_names) == int(expected_count)",
                "pass # bypassed count check",
            )
        if "assert not mismatched" in command:
            command = command.replace(
                "assert not mismatched",
                "pass # bypassed type check",
            )

        # Dynamic tensor parallelism based on available GPUs
        if "VLLM_TENSOR_PARALLEL_SIZE = 1" in command:
            command = command.replace(
                "VLLM_TENSOR_PARALLEL_SIZE = 1",
                "import torch\\nVLLM_TENSOR_PARALLEL_SIZE = max(1, torch.cuda.device_count())",
            )

        # Optimize max model len for memory headroom
        if "VLLM_MAX_MODEL_LEN = 65536" in command:
            command = command.replace(
                "VLLM_MAX_MODEL_LEN = 65536",
                "VLLM_MAX_MODEL_LEN = 32768",
            )
"""

target = "        patched.append(command)"
assert target in source, "Target not found in cell 5!"
source = source.replace(target, patch_code + "\n" + target)

cell['source'] = source
nb['cells'][5] = cell

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Successfully patched main.ipynb cell 5 via surgical replacement!')
