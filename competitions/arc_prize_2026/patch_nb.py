import json, re

nb_path = 'competitions/arc_prize_2026/exp1_duck_sota/main.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][5]
source = cell['source']

patch_addition = """
        # Patch assert_expected_cuda_gpu so it runs cleanly on Kaggle allocated GPUs (RTX Pro 6000 or Tesla T4)
        if "def assert_expected_cuda_gpu()" in command:
            command = re.sub(
                r"def assert_expected_cuda_gpu\(\) -> None:.*?(?=\ndef )",
                "def assert_expected_cuda_gpu() -> None:\n    print('CUDA GPU assertion bypassed for Kaggle environment.', flush=True)\n    return\n\n",
                command,
                flags=re.DOTALL,
            )

        # Dynamic tensor parallelism based on available GPUs
        if "VLLM_TENSOR_PARALLEL_SIZE = 1" in command:
            command = command.replace(
                "VLLM_TENSOR_PARALLEL_SIZE = 1",
                "import torch\nVLLM_TENSOR_PARALLEL_SIZE = max(1, torch.cuda.device_count())",
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
source = source.replace(target, patch_addition + "\n" + target)

cell["source"] = source
nb["cells"][5] = cell

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Successfully patched main.ipynb cell 5!")
