import json
from pathlib import Path

bundle_dir = Path("competitions/gemma_4_developer_agent/exp7_sequential_pipeline/agent_bundle")

sampling_yaml = (bundle_dir / "configs" / "sampling.yaml").read_text(encoding="utf-8")
eval_yaml = (bundle_dir / "eval_config.yaml").read_text(encoding="utf-8")
localizer_md = (bundle_dir / "prompts" / "localizer.md").read_text(encoding="utf-8")
reproducer_md = (bundle_dir / "prompts" / "reproducer.md").read_text(encoding="utf-8")
patcher_md = (bundle_dir / "prompts" / "patcher.md").read_text(encoding="utf-8")
finalizer_md = (bundle_dir / "prompts" / "finalizer.md").read_text(encoding="utf-8")

localizer_yaml = (bundle_dir / "sub_agents" / "localizer.yaml").read_text(encoding="utf-8")
reproducer_yaml = (bundle_dir / "sub_agents" / "reproducer.yaml").read_text(encoding="utf-8")
patcher_yaml = (bundle_dir / "sub_agents" / "patcher.yaml").read_text(encoding="utf-8")
finalizer_yaml = (bundle_dir / "sub_agents" / "finalizer.yaml").read_text(encoding="utf-8")
agent_yaml = (bundle_dir / "agent.yaml").read_text(encoding="utf-8")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Autobot Exp 7: 4-Stage Sequential SWE Pipeline with Isolated Contexts\n",
            "## Architecture: Localizer -> Reproducer -> Patcher -> Finalizer\n",
            "### SOTA Principles: Staged isolation (`include_contents: none`), unique anchors, guarded in-place fallback, guaranteed submission.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, shutil, zipfile, hashlib, json\n",
            "from pathlib import Path\n",
            "\n",
            "WORKING_DIR = Path('/kaggle/working').resolve()\n",
            "AGENT_DIR = WORKING_DIR / 'submission'\n",
            "if AGENT_DIR.exists():\n",
            "    shutil.rmtree(AGENT_DIR)\n",
            "AGENT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "(AGENT_DIR / 'configs').mkdir(parents=True, exist_ok=True)\n",
            "(AGENT_DIR / 'prompts').mkdir(parents=True, exist_ok=True)\n",
            "(AGENT_DIR / 'sub_agents').mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(f'Initialized build directory at: {AGENT_DIR}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Configs & Budgets\n",
            f"SAMPLING_YAML = {repr(sampling_yaml)}\n",
            "(AGENT_DIR / 'configs' / 'sampling.yaml').write_text(SAMPLING_YAML.strip() + '\\n', encoding='utf-8')\n",
            f"EVAL_YAML = {repr(eval_yaml)}\n",
            "(AGENT_DIR / 'eval_config.yaml').write_text(EVAL_YAML.strip() + '\\n', encoding='utf-8')\n",
            "\n",
            "# 2. Prompts\n",
            f"LOCALIZER_MD = {repr(localizer_md)}\n",
            "(AGENT_DIR / 'prompts' / 'localizer.md').write_text(LOCALIZER_MD.strip() + '\\n', encoding='utf-8')\n",
            f"REPRODUCER_MD = {repr(reproducer_md)}\n",
            "(AGENT_DIR / 'prompts' / 'reproducer.md').write_text(REPRODUCER_MD.strip() + '\\n', encoding='utf-8')\n",
            f"PATCHER_MD = {repr(patcher_md)}\n",
            "(AGENT_DIR / 'prompts' / 'patcher.md').write_text(PATCHER_MD.strip() + '\\n', encoding='utf-8')\n",
            f"FINALIZER_MD = {repr(finalizer_md)}\n",
            "(AGENT_DIR / 'prompts' / 'finalizer.md').write_text(FINALIZER_MD.strip() + '\\n', encoding='utf-8')\n",
            "\n",
            "# 3. Sub-Agent Declarations\n",
            f"LOCALIZER_YAML = {repr(localizer_yaml)}\n",
            "(AGENT_DIR / 'sub_agents' / 'localizer.yaml').write_text(LOCALIZER_YAML.strip() + '\\n', encoding='utf-8')\n",
            f"REPRODUCER_YAML = {repr(reproducer_yaml)}\n",
            "(AGENT_DIR / 'sub_agents' / 'reproducer.yaml').write_text(REPRODUCER_YAML.strip() + '\\n', encoding='utf-8')\n",
            f"PATCHER_YAML = {repr(patcher_yaml)}\n",
            "(AGENT_DIR / 'sub_agents' / 'patcher.yaml').write_text(PATCHER_YAML.strip() + '\\n', encoding='utf-8')\n",
            f"FINALIZER_YAML = {repr(finalizer_yaml)}\n",
            "(AGENT_DIR / 'sub_agents' / 'finalizer.yaml').write_text(FINALIZER_YAML.strip() + '\\n', encoding='utf-8')\n",
            "\n",
            "# 4. Root Sequential Pipeline\n",
            f"AGENT_YAML = {repr(agent_yaml)}\n",
            "(AGENT_DIR / 'agent.yaml').write_text(AGENT_YAML.strip() + '\\n', encoding='utf-8')\n",
            "\n",
            "print('Wrote all 11 pipeline configuration and prompt files successfully.')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Package submission.zip with deterministic timestamps\n",
            "ZIP_PATH = WORKING_DIR / 'submission.zip'\n",
            "ZIP_PATH.unlink(missing_ok=True)\n",
            "\n",
            "with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
            "    for p in sorted(p for p in AGENT_DIR.rglob('*') if p.is_file()):\n",
            "        rel_path = str(p.relative_to(AGENT_DIR)).replace('\\\\', '/')\n",
            "        info = zipfile.ZipInfo(rel_path, date_time=(1980, 1, 1, 0, 0, 0))\n",
            "        info.compress_type = zipfile.ZIP_DEFLATED\n",
            "        info.external_attr = 0o644 << 16\n",
            "        zf.writestr(info, p.read_bytes())\n",
            "\n",
            "size_kb = ZIP_PATH.stat().st_size / 1024.0\n",
            "print(f'Successfully created {ZIP_PATH.name} ({size_kb:.2f} KB)')\n",
            "with zipfile.ZipFile(ZIP_PATH, 'r') as zf:\n",
            "    for name in zf.namelist():\n",
            "        print(f'  - {name}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Rigorous validation against swegemma and adk-submission limits\n",
            "import yaml\n",
            "\n",
            "class SafeIncludeLoader(yaml.SafeLoader):\n",
            "    pass\n",
            "SafeIncludeLoader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))\n",
            "\n",
            "with open(AGENT_DIR / 'agent.yaml', 'r', encoding='utf-8') as f:\n",
            "    root_cfg = yaml.load(f, Loader=SafeIncludeLoader)\n",
            "assert root_cfg['agent_class'] == 'SequentialAgent', 'Root must be SequentialAgent'\n",
            "\n",
            "models = set()\n",
            "for sub in root_cfg['sub_agents']:\n",
            "    sub_p = (AGENT_DIR / sub['config_path']).resolve()\n",
            "    assert sub_p.exists(), f'Missing sub-agent file: {sub[\"config_path\"]}'\n",
            "    with open(sub_p, 'r', encoding='utf-8') as f:\n",
            "        sub_c = yaml.load(f, Loader=SafeIncludeLoader)\n",
            "    models.add(sub_c['model'])\n",
            "\n",
            "assert models == {'gemma-4-31b-it-qat-w4a16-ct'}, f'Single base model violation: {models}'\n",
            "assert ZIP_PATH.stat().st_size < 3 * 1024 * 1024 * 1024, 'Archive exceeds 3 GiB'\n",
            "\n",
            "print('\\n======================================================')\n",
            "print('SUCCESS: EXP 7 SEQUENTIAL PIPELINE VALIDATION PASSED!')\n",
            "print(f'Submission: {ZIP_PATH} ({size_kb:.2f} KB) ready for scoring.')\n",
            "print('======================================================')\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_nb = Path("competitions/gemma_4_developer_agent/exp7_sequential_pipeline/main.ipynb")
out_nb.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(f"Successfully wrote notebook to {out_nb}")
