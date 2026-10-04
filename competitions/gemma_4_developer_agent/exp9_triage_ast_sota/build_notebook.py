"""
Builder script to generate competitions/gemma_4_developer_agent/exp9_triage_ast_sota/main.ipynb
with zero AST errors and complete embedded agent bundle.
"""

import json
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT_BUNDLE_DIR = HERE / "agent_bundle"


def load_file_content(rel_path: str) -> str:
    path = AGENT_BUNDLE_DIR / rel_path
    return path.read_text(encoding="utf-8")


def build_exp9_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Cell 0: Markdown Header
    cell0_md = """# Autobot Gemma 4 Developer Agent · Experiment 9 SOTA
## Autonomous Software Engineering Agent for the Gemma 4 Developer Agent Challenge ($65,000 USD)
### Architecture: Dynamic Problem Triage + First-Turn AST Grounding + Ephemeral `/tmp` Repro + Resilient Context Anchoring

**Key Upgrades in Experiment 9**:
1. **Dynamic Problem Triage & Fast-Exit**: Turn-1 complexity classifier. Bails in 15 seconds on intractable multi-package refactors; preserves deep 4.5-minute budgets for high-probability localized defects.
2. **First-Turn AST Graph Tool Protocol**: Targets `search_similar_code(symbol)`, `get_code_neighbors(node)`, and `get_code_subgraph(nodes)`. Replaces blind 10-call grep loops with 1-call exact symbol resolution.
3. **Ephemeral `/tmp/repro.py` Verification**: Isolated reproduction script in `/tmp/` guarantees clean `/workspace` git diffs while proving bug reproduction and fix validity.
4. **Resilient 4-to-6 Line Context Anchoring**: Eliminates `FileEditError: Multiple occurrences found` with wide anchor windows and whole-function replacement fallback.
5. **Strict Test Anti-Tampering Immunity**: Guarantees zero edits to `tests/`, `conftest.py`, or `pytest.ini`, preventing Container B automatic test rollback failures.
"""
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": cell0_md.splitlines(keepends=True)})

    # Cell 1: Environment Setup
    cell1_code = '''import os
import sys
import shutil
import zipfile
import hashlib
import json
from pathlib import Path

WORKING_DIR = Path('/kaggle/working').resolve()
AGENT_DIR = WORKING_DIR / 'submission'
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR)

AGENT_DIR.mkdir(parents=True, exist_ok=True)
(AGENT_DIR / 'configs').mkdir(parents=True, exist_ok=True)
(AGENT_DIR / 'prompts').mkdir(parents=True, exist_ok=True)
(AGENT_DIR / 'sub_agents').mkdir(parents=True, exist_ok=True)

print(f"[INIT] Initialized submission build directory at: {AGENT_DIR}")
'''
    ast.parse(cell1_code)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1_code.splitlines(keepends=True)})

    # Cell 2: Write Declarative Files
    sampling_yaml = load_file_content("configs/sampling.yaml")
    eval_yaml = load_file_content("eval_config.yaml")
    analyzer_md = load_file_content("prompts/analyzer.md")
    system_md = load_file_content("prompts/system.md")
    analyzer_yaml = load_file_content("sub_agents/code_analyzer.yaml")
    agent_yaml = load_file_content("agent.yaml")

    cell2_code = f'''# 1. Write configs/sampling.yaml
SAMPLING_YAML = {json.dumps(sampling_yaml)}
(AGENT_DIR / 'configs' / 'sampling.yaml').write_text(SAMPLING_YAML.strip() + '\\n', encoding='utf-8')

# 2. Write configs/eval_config.yaml
EVAL_YAML = {json.dumps(eval_yaml)}
(AGENT_DIR / 'eval_config.yaml').write_text(EVAL_YAML.strip() + '\\n', encoding='utf-8')

# 3. Write prompts/analyzer.md
ANALYZER_MD = {json.dumps(analyzer_md)}
(AGENT_DIR / 'prompts' / 'analyzer.md').write_text(ANALYZER_MD.strip() + '\\n', encoding='utf-8')

# 4. Write prompts/system.md
SYSTEM_MD = {json.dumps(system_md)}
(AGENT_DIR / 'prompts' / 'system.md').write_text(SYSTEM_MD.strip() + '\\n', encoding='utf-8')

# 5. Write sub_agents/code_analyzer.yaml
CODE_ANALYZER_YAML = {json.dumps(analyzer_yaml)}
(AGENT_DIR / 'sub_agents' / 'code_analyzer.yaml').write_text(CODE_ANALYZER_YAML.strip() + '\\n', encoding='utf-8')

# 6. Write root agent.yaml
AGENT_YAML = {json.dumps(agent_yaml)}
(AGENT_DIR / 'agent.yaml').write_text(AGENT_YAML.strip() + '\\n', encoding='utf-8')

print("[SUCCESS] Wrote all 6 declarative configuration and prompt files to submission bundle.")
'''
    ast.parse(cell2_code)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell2_code.splitlines(keepends=True)})

    # Cell 3: Package Zip Archive
    cell3_code = '''# Package submission.zip with deterministic timestamps and POSIX file attributes
ZIP_PATH = WORKING_DIR / 'submission.zip'
ZIP_PATH.unlink(missing_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(p for p in AGENT_DIR.rglob('*') if p.is_file()):
        rel_path = str(p.relative_to(AGENT_DIR)).replace('\\\\', '/')
        info = zipfile.ZipInfo(rel_path, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, p.read_bytes())

size_kb = ZIP_PATH.stat().st_size / 1024.0
print(f"[PACKAGE] Successfully generated {ZIP_PATH.name} ({size_kb:.2f} KB)")
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    for name in zf.namelist():
        print(f"  - {name}")
'''
    ast.parse(cell3_code)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell3_code.splitlines(keepends=True)})

    # Cell 4: Rigorous ADK Validation
    cell4_code = '''import yaml

# Safe custom YAML loader that handles Google ADK !include tags
class SafeIncludeLoader(yaml.SafeLoader):
    pass

def include_constructor(loader, node):
    return f"!include {loader.construct_scalar(node)}"

SafeIncludeLoader.add_constructor('!include', include_constructor)

print("=== AUTOBOT SWE-GEMMA SUBMISSION VALIDATION SUITE ===")

# Gate 1: Archive Size Check (< 3 GiB limit)
archive_bytes = ZIP_PATH.stat().st_size
assert archive_bytes < 3 * 1024 * 1024 * 1024, f"Archive exceeds 3 GiB: {archive_bytes} bytes"
print(f"Gate 1 Passed: Archive size {archive_bytes / 1024:.2f} KB is well within 3 GiB ceiling.")

# Gate 2: Required Root Config Discovery
root_agent_path = AGENT_DIR / 'agent.yaml'
assert root_agent_path.is_file(), "Missing required root agent.yaml"
print("Gate 2 Passed: Root agent.yaml discovered at submission root.")

# Gate 3: Parse and Validate Root Agent YAML
with open(root_agent_path, 'r', encoding='utf-8') as f:
    root_agent_data = yaml.load(f, Loader=SafeIncludeLoader)

assert root_agent_data.get('name') == 'autobot_swe_coder', f"Unexpected agent name: {root_agent_data.get('name')}"
print(f"Gate 3 Passed: Root agent verified: {root_agent_data.get('name')}")

# Gate 4: Single Base Model Rule (Hard Competition Constraint)
ALLOWED_MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
assert root_agent_data.get('model') == ALLOWED_MODEL, f"Invalid root model: {root_agent_data.get('model')}"

analyzer_path = AGENT_DIR / 'sub_agents' / 'code_analyzer.yaml'
assert analyzer_path.is_file(), "Missing sub_agents/code_analyzer.yaml"
with open(analyzer_path, 'r', encoding='utf-8') as f:
    analyzer_data = yaml.load(f, Loader=SafeIncludeLoader)
assert analyzer_data.get('model') == ALLOWED_MODEL, f"Invalid analyzer model: {analyzer_data.get('model')}"
print(f"Gate 4 Passed: Single Base Model verified strictly as '{ALLOWED_MODEL}' across all agents.")

# Gate 5: Tool Binding & Declarative Schema Check
root_tools = root_agent_data.get('tools', [])
assert 'submit_patch' in root_tools, "Root agent missing mandatory submit_patch tool!"
assert 'edit_file' in root_tools, "Root agent missing edit_file tool!"
assert 'read_file' in root_tools, "Root agent missing read_file tool!"
assert 'run_command' in root_tools, "Root agent missing run_command tool!"
print(f"Gate 5 Passed: All required workspace tools declared ({len(root_tools)} tools bound).")

# Gate 6: Operational Budget Limits (eval_config.yaml)
eval_cfg_path = AGENT_DIR / 'eval_config.yaml'
assert eval_cfg_path.is_file(), "Missing eval_config.yaml"
with open(eval_cfg_path, 'r', encoding='utf-8') as f:
    eval_cfg = yaml.safe_load(f).get('evaluation', {})

assert eval_cfg.get('max_time_minutes', 0) <= 60.0, f"Exceeds max time minutes: {eval_cfg.get('max_time_minutes')}"
assert eval_cfg.get('max_tool_calls', 0) <= 100, f"Exceeds max tool calls: {eval_cfg.get('max_tool_calls')}"
assert eval_cfg.get('timeout_seconds', 0) <= 300, f"Exceeds command timeout: {eval_cfg.get('timeout_seconds')}"
print(f"Gate 6 Passed: Operational budgets verified: {eval_cfg['max_time_minutes']}m / {eval_cfg['max_tool_calls']} tools / {eval_cfg['timeout_seconds']}s timeout.")

# Gate 7: Sampling & Thinking Configuration (configs/sampling.yaml)
sampling_path = AGENT_DIR / 'configs' / 'sampling.yaml'
assert sampling_path.is_file(), "Missing configs/sampling.yaml"
with open(sampling_path, 'r', encoding='utf-8') as f:
    sampling_data = yaml.safe_load(f)

thinking_cfg = sampling_data.get('thinking_config', {})
assert thinking_cfg.get('include_thoughts') is True, "Thinking must be enabled for SWE-bench reasoning!"
assert 0 < thinking_cfg.get('thinking_budget', 0) <= 32768, f"Invalid thinking budget: {thinking_cfg.get('thinking_budget')}"
print(f"Gate 7 Passed: Gemma 4 thinking engine verified: budget={thinking_cfg['thinking_budget']} tokens.")

# Gate 8: Checksum and Artifact Audit
sha256_hash = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
print(f"Gate 8 Passed: submission.zip SHA256: {sha256_hash}")

print("\\n" + "=" * 65)
print("ALL 8 ADK COMPETITION VALIDATION GATES PASSED CLEANLY (100% READY FOR SUBMISSION)")
print("=" * 65)
'''
    ast.parse(cell4_code)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell4_code.splitlines(keepends=True)})

    # Write target notebook
    target_nb = HERE / "main.ipynb"
    with open(target_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"[SUCCESS] Authored {target_nb} ({len(nb['cells'])} cells) with zero AST errors.")


if __name__ == "__main__":
    build_exp9_notebook()
