import json
from pathlib import Path

bundle_dir = Path("competitions/gemma_4_developer_agent/exp6_autobot_swe_sota/agent_bundle")

sampling_yaml = (bundle_dir / "configs" / "sampling.yaml").read_text(encoding="utf-8")
eval_yaml = (bundle_dir / "eval_config.yaml").read_text(encoding="utf-8")
analyzer_md = (bundle_dir / "prompts" / "analyzer.md").read_text(encoding="utf-8")
system_md = (bundle_dir / "prompts" / "system.md").read_text(encoding="utf-8")
code_analyzer_yaml = (bundle_dir / "sub_agents" / "code_analyzer.yaml").read_text(encoding="utf-8")
agent_yaml = (bundle_dir / "agent.yaml").read_text(encoding="utf-8")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Autobot Gemma 4 SWE-Coder SOTA\n",
            "## Autonomous Software Engineering Agent for the Gemma 4 Developer Agent Challenge\n",
            "### Architecture: Dual-Agent Specialization (`code_analyzer` + `swe_coder`) with Enabled Gemma 4 Thinking\n"
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
            "# 1. Write configs/sampling.yaml with enabled Gemma 4 chain-of-thought thinking\n",
            f"SAMPLING_YAML = {repr(sampling_yaml)}\n",
            "(AGENT_DIR / 'configs' / 'sampling.yaml').write_text(SAMPLING_YAML.strip() + '\\n', encoding='utf-8')\n",
            "\n",
            "# 2. Write eval_config.yaml with calibrated SWE budgets\n",
            f"EVAL_YAML = {repr(eval_yaml)}\n",
            "(AGENT_DIR / 'eval_config.yaml').write_text(EVAL_YAML.strip() + '\\n', encoding='utf-8')\n",
            "\n",
            "# 3. Write prompts/analyzer.md for root-cause localization\n",
            f"ANALYZER_MD = {repr(analyzer_md)}\n",
            "(AGENT_DIR / 'prompts' / 'analyzer.md').write_text(ANALYZER_MD.strip() + '\\n', encoding='utf-8')\n",
            "\n",
            "# 4. Write prompts/system.md for root SWE-Coder\n",
            f"SYSTEM_MD = {repr(system_md)}\n",
            "(AGENT_DIR / 'prompts' / 'system.md').write_text(SYSTEM_MD.strip() + '\\n', encoding='utf-8')\n",
            "\n",
            "# 5. Write sub_agents/code_analyzer.yaml declarative agent\n",
            f"CODE_ANALYZER_YAML = {repr(code_analyzer_yaml)}\n",
            "(AGENT_DIR / 'sub_agents' / 'code_analyzer.yaml').write_text(CODE_ANALYZER_YAML.strip() + '\\n', encoding='utf-8')\n",
            "\n",
            "# 6. Write root agent.yaml\n",
            f"AGENT_YAML = {repr(agent_yaml)}\n",
            "(AGENT_DIR / 'agent.yaml').write_text(AGENT_YAML.strip() + '\\n', encoding='utf-8')\n",
            "\n",
            "print('Wrote all 6 declarative agent configuration and prompt files successfully.')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Package submission.zip with deterministic timestamps\n",
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
            "# 8. Rigorous validation against swegemma and adk-submission limits\n",
            "import yaml\n",
            "\n",
            "class SafeIncludeLoader(yaml.SafeLoader):\n",
            "    pass\n",
            "SafeIncludeLoader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))\n",
            "\n",
            "# Verify root agent\n",
            "with open(AGENT_DIR / 'agent.yaml', 'r', encoding='utf-8') as f:\n",
            "    root_cfg = yaml.load(f, Loader=SafeIncludeLoader)\n",
            "assert root_cfg['model'] == 'gemma-4-31b-it-qat-w4a16-ct', f\"Invalid base model: {root_cfg['model']}\"\n",
            "\n",
            "# Verify code analyzer\n",
            "with open(AGENT_DIR / 'sub_agents' / 'code_analyzer.yaml', 'r', encoding='utf-8') as f:\n",
            "    sub_cfg = yaml.load(f, Loader=SafeIncludeLoader)\n",
            "assert sub_cfg['model'] == 'gemma-4-31b-it-qat-w4a16-ct', f\"Invalid sub-agent model: {sub_cfg['model']}\"\n",
            "\n",
            "# Verify submission size\n",
            "MAX_SUBMISSION_BYTES = 3 * 1024 * 1024 * 1024\n",
            "assert ZIP_PATH.stat().st_size < MAX_SUBMISSION_BYTES, 'Submission exceeds 3 GiB limit'\n",
            "\n",
            "print('\\n=========================================')\n",
            "print('✅ ALL AUTOBOT SUBMISSION CHECKS PASSED!')\n",
            "print(f'Artifact: {ZIP_PATH} is ready for scoring!')\n",
            "print('=========================================')\n"
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

out_nb = Path("competitions/gemma_4_developer_agent/exp6_autobot_swe_sota/main.ipynb")
out_nb.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(f"Successfully wrote notebook to {out_nb}")
