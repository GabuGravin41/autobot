"""
Exp 5 -- Evaluate exp4's LoRA adapter against the SAME approximate local
harness and SAME ~10-task sample exp3 used, for an apples-to-apples
zero-shot-vs-LoRA comparison. Everything except model loading (base model
+ PeftModel adapter wrap) is copied verbatim from
exp3_ast_call_parsing/main.py -- same TOOL_DOCS, same parser, same
Sandbox, same verify_task, same task-sampling code (which is fully
deterministic given the same tasks.jsonl file order, so this reproduces
exp3's exact 10-task sample without needing to hardcode instance_ids).

WHY THIS EXISTS: see exp3's module docstring / THINKING_AND_DECISIONS.md
section 3 -- the real swegemma/adk-submission/adk-eval-core harness isn't
locally runnable, so this is our own best-effort reimplementation, used
ONLY to get a directional zero-shot-vs-LoRA signal before spending any of
the competition's 2 allowed submissions. Its pass rate is NOT the real
score.

Runs entirely on Kaggle (GPU kernel). Writes /kaggle/working/results/.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Config (identical to exp3, so the task sample matches exactly)
# ---------------------------------------------------------------------------
N_TASKS = int(os.environ.get("N_TASKS", "10"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "16"))
MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "16"))
COMMAND_TIMEOUT = 120
MAX_STDOUT_CHARS = 5000
MAX_FILE_LINES = 150
MAX_FILE_CHARS = 10000
MAX_NEW_TOKENS = 700

WORK_ROOT = Path("/kaggle/working")
RESULTS_DIR = WORK_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PATCHES_DIR = RESULTS_DIR / "patches"
for d in (RESULTS_DIR, LOGS_DIR, PATCHES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Locate competition data + base model + LoRA adapter on this Kaggle kernel
# ---------------------------------------------------------------------------
def find_dir(candidates_globs):
    for pattern in candidates_globs:
        for hit in sorted(glob.glob(pattern, recursive=True)):
            p = Path(hit)
            if p.is_dir():
                return p
    return None


def find_file(candidates_globs):
    for pattern in candidates_globs:
        for hit in sorted(glob.glob(pattern, recursive=True)):
            if Path(hit).is_file():
                return Path(hit)
    return None


DATA_DIR = find_dir([
    "/kaggle/input/gemma-4-developer-agent",
    "/kaggle/input/competitions/gemma-4-developer-agent",
    "/kaggle/input/*gemma*developer*agent*",
])
MODEL_DIR = find_dir([
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/*",
    "/kaggle/input/models/google/gemma-4/*/gemma-4-e4b-it/*",
    "/kaggle/input/gemma-4/transformers/gemma-4-e4b-it/*",
    "/kaggle/input/gemma-4/*/gemma-4-e4b-it/*",
    "/kaggle/input/**/gemma-4-e4b-it/*",
])
# exp4's adapter, mounted via kernel_sources (its /kaggle/working/adapter/
# becomes /kaggle/input/autobot-gemma4-exp4-lora-sft/adapter/ here).
_adapter_config = find_file([
    "/kaggle/input/*exp4-lora-sft*/adapter/adapter_config.json",
    "/kaggle/input/**/adapter/adapter_config.json",
    "/kaggle/input/**/adapter_config.json",
])
ADAPTER_DIR = _adapter_config.parent if _adapter_config else None

subprocess.run(["git", "config", "--global", "user.email", "autobot@local"], capture_output=True)
subprocess.run(["git", "config", "--global", "user.name", "autobot"], capture_output=True)
subprocess.run(["git", "config", "--global", "safe.directory", "*"], capture_output=True)

log(f"DATA_DIR = {DATA_DIR}")
log(f"MODEL_DIR = {MODEL_DIR}")
log(f"ADAPTER_DIR = {ADAPTER_DIR}")
if DATA_DIR is None or MODEL_DIR is None or ADAPTER_DIR is None:
    log("FATAL: could not resolve data, model, or adapter directory. Listing /kaggle/input for diagnosis:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
        if depth > 6:
            dirs[:] = []
            continue
        log(f"  {root}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load tasks.jsonl, take the SAME small stratified sample exp3 used
# (deterministic given the same file order -- no hardcoded instance_ids
# needed to reproduce it).
# ---------------------------------------------------------------------------
tasks = []
with open(DATA_DIR / "tasks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
log(f"Loaded {len(tasks)} total public tasks")

by_repo = {}
for t in tasks:
    by_repo.setdefault(t["repo"], []).append(t)

sample = []
repos = sorted(by_repo.keys())
i = 0
while len(sample) < min(N_TASKS, len(tasks)) and repos:
    repo = repos[i % len(repos)]
    if by_repo[repo]:
        sample.append(by_repo[repo].pop(0))
    else:
        repos.remove(repo)
        continue
    i += 1
log(f"Sampled {len(sample)} tasks (same deterministic sample as exp3): {[t['instance_id'] for t in sample]}")

# ---------------------------------------------------------------------------
# Load base model (4-bit NF4, same as exp3) + wrap with exp4's LoRA adapter
# ---------------------------------------------------------------------------
for pkg in ("transformers", "accelerate", "bitsandbytes", "peft"):
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", pkg],
                            capture_output=True, text=True, timeout=240)
        log(f"pip install -U {pkg}: exit={r.returncode}" + (f" stderr_tail={r.stderr[-300:]}" if r.returncode else ""))
    except Exception as e:
        log(f"pip install -U {pkg} failed: {e}")

log("Loading model...")
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
log(f"transformers version: {__import__('transformers').__version__}")
log(f"peft version: {__import__('peft').__version__}")

try:
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    tokenizer = getattr(processor, "tokenizer", processor)
except Exception as e:
    log(f"AutoProcessor failed ({e}), falling back to AutoTokenizer")
    processor = None
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
log("Base model loaded. Applying LoRA adapter from exp4...")
model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
model.eval()
log("Adapter applied.")


def _apply_chat_template(messages):
    """Render `messages`, folding system->first-user-turn if the chat
    template rejects a bare system role (varies across Gemma releases).
    Identical to exp3's helper."""
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    target = processor if processor is not None else tokenizer
    if processor is not None:
        kwargs["enable_thinking"] = False
    try:
        return target.apply_chat_template(messages, **kwargs)
    except Exception:
        folded = []
        sys_text = None
        for m in messages:
            if m["role"] == "system":
                sys_text = m["content"]
                continue
            if sys_text and m["role"] == "user" and not folded:
                folded.append({"role": "user", "content": f"{sys_text}\n\n{m['content']}"})
                sys_text = None
            else:
                folded.append(m)
        return target.apply_chat_template(folded, **kwargs)


def generate(messages):
    prompt = _apply_chat_template(messages)
    if processor is not None:
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    decode_fn = processor.decode if processor is not None else tokenizer.decode
    text = decode_fn(new_tokens, skip_special_tokens=True)
    if processor is not None and hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(text)
            if isinstance(parsed, dict) and parsed.get("content"):
                return parsed["content"]
        except Exception:
            pass
    return text


# ---------------------------------------------------------------------------
# Approximate sandbox tools -- identical to exp3
# (HARNESS_README.md section 6 reimplementation, NOT the real grader sandbox)
# ---------------------------------------------------------------------------
class Sandbox:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.tool_calls_used = 0
        self.patch_submitted = False
        self.submitted_patch = ""

    def _resolve(self, filepath: str) -> Path:
        filepath = filepath.lstrip("/")
        if filepath.startswith("workspace/"):
            filepath = filepath[len("workspace/"):]
        if ".." in Path(filepath).parts:
            raise ValueError("path traversal blocked")
        return self.workspace / filepath

    def run_command(self, command: str) -> dict:
        self.tool_calls_used += 1
        try:
            r = subprocess.run(
                ["bash", "-c", command],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT,
            )
            stdout = r.stdout[:MAX_STDOUT_CHARS]
            stderr = r.stderr[:MAX_STDOUT_CHARS]
            if r.returncode == 0:
                return {"status": "ok", "stdout": stdout, "exit_code": 0}
            return {"status": "error", "error_type": "CommandError",
                     "error_message": f"Command failed with exit code {r.returncode}",
                     "details": {"stdout": stdout, "stderr": stderr, "exit_code": r.returncode}}
        except subprocess.TimeoutExpired:
            return {"status": "error", "error_type": "TimeoutExceeded", "error_message": "Command timed out"}

    def read_file(self, filepath: str, start_line=None, end_line=None) -> dict:
        self.tool_calls_used += 1
        p = self._resolve(filepath)
        if not p.exists():
            return {"status": "error", "error_type": "FileNotFound", "error_message": f"{filepath} does not exist"}
        if p.is_dir():
            entries = sorted(x.name for x in p.iterdir())[:50]
            return {"status": "error", "error_type": "IsADirectory",
                     "error_message": f"{filepath} is a directory, not a file", "details": {"entries": entries}}
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        total = len(lines)
        start = (start_line - 1) if start_line else 0
        end = end_line if end_line else total
        chunk = lines[start:end]
        is_truncated = False
        if len(chunk) > MAX_FILE_LINES:
            chunk = chunk[:MAX_FILE_LINES]
            is_truncated = True
        content = "".join(chunk)
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS]
            is_truncated = True
        return {"status": "ok", "filepath": filepath, "content": content,
                "start_line": start + 1, "end_line": start + len(chunk), "total_lines": total,
                "is_truncated": is_truncated}

    def edit_file(self, filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> dict:
        self.tool_calls_used += 1
        p = self._resolve(filepath)
        if not p.exists() or p.stat().st_size == 0:
            return {"status": "error", "error_type": "FileEditError", "error_message": "file missing or empty"}
        if not old_string:
            return {"status": "error", "error_type": "FileEditError", "error_message": "old_string is empty"}
        content = p.read_text(encoding="utf-8", errors="replace")

        count = content.count(old_string)
        if count >= 1:
            if count > 1 and not allow_multiple:
                return {"status": "error", "error_type": "FileEditError",
                         "error_message": f"old_string matched {count} times (must be unique)"}
            p.write_text(content.replace(old_string, new_string, 1 if not allow_multiple else -1), encoding="utf-8")
            return {"status": "ok", "filepath": filepath, "occurrences": count, "strategy": "exact"}

        old_lines = [l.strip() for l in old_string.splitlines() if l.strip()]
        content_lines = content.splitlines()
        for i in range(len(content_lines) - len(old_lines) + 1):
            window = [content_lines[i + j].strip() for j in range(len(old_lines))]
            if window == old_lines:
                indent = len(content_lines[i]) - len(content_lines[i].lstrip())
                indented_new = "\n".join((" " * indent + l if l else "") for l in new_string.splitlines())
                new_lines = content_lines[:i] + [indented_new] + content_lines[i + len(old_lines):]
                p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                return {"status": "ok", "filepath": filepath, "occurrences": 1, "strategy": "flexible"}

        return {"status": "error", "error_type": "FileEditError",
                 "error_message": "old_string not found via exact or flexible matching"}

    def write_file(self, filepath: str, content: str) -> dict:
        self.tool_calls_used += 1
        p = self._resolve(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"status": "ok", "filepath": filepath, "size": len(content)}

    def get_status(self) -> dict:
        return {"tool_calls_used": self.tool_calls_used, "patch_submitted": self.patch_submitted,
                 "max_tool_calls": MAX_TOOL_CALLS}

    def submit_patch(self) -> dict:
        subprocess.run(["git", "add", "-N", "."], cwd=self.workspace, capture_output=True)
        r = subprocess.run(["git", "diff", "HEAD"], cwd=self.workspace, capture_output=True, text=True)
        self.submitted_patch = r.stdout
        self.patch_submitted = True
        return {"status": "ok", "patch_size": len(r.stdout)}


TOOL_DOCS = """Available tools -- call exactly ONE per turn by responding with a single fenced block:
```tool_call
{"tool": "<name>", "args": {"argname": "value", ...}}
```
"args" must be a JSON object with the tool's named arguments (e.g. {"filepath": "x.py"}), not a list and not flattened
directly into the top-level object. Tools:
- run_command(command): run a shell command in /workspace (bash -c).
- read_file(filepath, start_line=None, end_line=None): read a file (max 150 lines / 10000 chars per call).
- edit_file(filepath, old_string, new_string, allow_multiple=False): replace old_string with new_string. old_string must be an exact, unique snippet from the file.
- write_file(filepath, content): create or overwrite a file.
- get_status(): check your remaining budget.
- submit_patch(): finalize your diff. Call this LAST, exactly once, after verifying your fix.
Do not modify /workspace/pytest.ini or /workspace/conftest.py. Put scratch/reproduction scripts in /tmp, not /workspace.
"""

TOOL_CALL_RE = re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL)
LOOSE_JSON_RE = re.compile(r"\{[^{}]*\"tool\"\s*:\s*\"[a-zA-Z_]+\"[^{}]*\}", re.DOTALL)
PSEUDO_CALL_RE = re.compile(r"call:\s*([a-zA-Z_]+)\((.*)\)\s*$", re.MULTILINE)


def parse_tool_call(text: str):
    m = TOOL_CALL_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    m = LOOSE_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    m = PSEUDO_CALL_RE.search(text)
    if m:
        name, raw_args = m.group(1), m.group(2).strip()
        if not raw_args:
            return {"tool": name, "args": {}}
        import ast
        try:
            call_node = ast.parse(f"f({raw_args})", mode="eval").body
            positional = [ast.literal_eval(a) for a in call_node.args]
            keywords = {kw.arg: ast.literal_eval(kw.value) for kw in call_node.keywords}
            if keywords and not positional:
                return {"tool": name, "args": keywords}
            if positional and not keywords:
                return {"tool": name, "args": positional}
            if positional or keywords:
                return {"tool": name, "args": positional, "kwargs": keywords}
            return {"tool": name, "args": {}}
        except Exception:
            return None
    return None


def dispatch(sandbox: Sandbox, call: dict):
    name = call.get("tool")
    fn = getattr(sandbox, name, None)
    if fn is None:
        return {"status": "error", "error_type": "UnknownTool", "error_message": f"no such tool: {name}"}

    args = call.get("args")
    kwargs = call.get("kwargs") or {}
    if args is None:
        args = {k: v for k, v in call.items() if k not in ("tool", "kwargs")}
    if isinstance(args, list):
        import inspect
        params = [p for p in inspect.signature(fn).parameters.keys() if p != "self"]
        args = dict(zip(params, args))
        args.update(kwargs)
    if not isinstance(args, dict):
        return {"status": "error", "error_type": "BadArgs", "error_message": f"unparseable args: {args!r}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"status": "error", "error_type": "BadArgs", "error_message": str(e)}


# ---------------------------------------------------------------------------
# Repo snapshot helpers -- identical to exp3
# ---------------------------------------------------------------------------
_LAYOUT_LOGGED = False


def extract_snapshot(instance_id: str, dest: Path):
    global _LAYOUT_LOGGED
    tgz = DATA_DIR / "snapshots" / f"{instance_id}.tgz"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz) as tf:
        tf.extractall(dest)
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and not (dest / ".git").exists():
        inner = entries[0]
        for item in inner.iterdir():
            shutil.move(str(item), str(dest / item.name))
        inner.rmdir()
    has_git = (dest / ".git").exists()
    if not _LAYOUT_LOGGED:
        _LAYOUT_LOGGED = True
        top = sorted(p.name for p in dest.iterdir())[:20]
        log(f"[layout diag] {instance_id}: top-level entries={top} has_.git={has_git}")
    if not has_git:
        subprocess.run(["git", "init", "-q"], cwd=dest, capture_output=True)
        subprocess.run(["git", "config", "user.email", "autobot@local"], cwd=dest, capture_output=True)
        subprocess.run(["git", "config", "user.name", "autobot"], cwd=dest, capture_output=True)
    r = subprocess.run(["git", "add", "-A"], cwd=dest, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"[layout diag] git add -A failed in {dest}: {r.stderr[:500]}")
    subprocess.run(["git", "commit", "--allow-empty", "-q", "-m", "baseline"], cwd=dest, capture_output=True)


def apply_patch_resilient(repo_dir: Path, patch_text: str) -> bool:
    if not patch_text.strip():
        return False
    patch_file = repo_dir / ".." / "_patch.diff"
    patch_file = patch_file.resolve()
    patch_file.write_text(patch_text, encoding="utf-8")
    attempts = [
        ["git", "apply", "--whitespace=fix", str(patch_file)],
        ["git", "apply", "-p0", "--whitespace=fix", str(patch_file)],
        ["patch", "-p1", "-i", str(patch_file)],
        ["patch", "-p0", "-i", str(patch_file)],
    ]
    for cmd in attempts:
        r = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        if r.returncode == 0:
            return True
    return False


def extract_test_paths(test_patch: str):
    return sorted(set(re.findall(r"^\+\+\+ b/(\S+)", test_patch, re.MULTILINE)))


def verify_task(instance_id: str, agent_patch: str, task: dict, scratch: Path):
    verify_dir = scratch / "verify"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    extract_snapshot(instance_id, verify_dir)

    applied = apply_patch_resilient(verify_dir, agent_patch)
    if not applied:
        return {"resolved": False, "reason": "agent_patch did not apply", "test_exit_code": None}

    test_paths = extract_test_paths(task["test_patch"])
    if not apply_patch_resilient(verify_dir, task["test_patch"]):
        return {"resolved": False, "reason": "test_patch did not apply (harness approximation gap)", "test_exit_code": None}

    if not test_paths:
        return {"resolved": False, "reason": "no test paths parsed from test_patch", "test_exit_code": None}

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", *test_paths],
        cwd=verify_dir, capture_output=True, text=True, timeout=300,
    )
    return {"resolved": r.returncode == 0, "reason": None, "test_exit_code": r.returncode,
             "test_stdout_tail": r.stdout[-2000:], "test_stderr_tail": r.stderr[-2000:]}


# ---------------------------------------------------------------------------
# Agent loop for one task -- identical to exp3
# ---------------------------------------------------------------------------
def run_agent_on_task(task: dict, scratch: Path):
    instance_id = task["instance_id"]
    log(f"=== {instance_id} ({task['repo']}) ===")
    workspace = scratch / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    extract_snapshot(instance_id, workspace)
    sandbox = Sandbox(workspace)

    hints = f"\n## Hints:\n{task['hints_text'].strip()}" if task.get("hints_text", "").strip() else ""
    system_prompt = (
        "You are an autonomous software engineering agent powered by Gemma 4. "
        "Resolve the described issue with a minimal, correct patch, then call submit_patch().\n\n" + TOOL_DOCS
    )
    user_prompt = (
        f"Repository: {task['repo']}\n\nProblem Statement:\n{task['problem_statement']}{hints}\n\n"
        f"Work inside /workspace. Investigate first (read_file), then make your fix (edit_file/write_file), "
        f"then call submit_patch()."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    transcript = []
    turn = 0
    while turn < MAX_TURNS and sandbox.tool_calls_used < MAX_TOOL_CALLS and not sandbox.patch_submitted:
        turn += 1
        try:
            reply = generate(messages)
        except Exception as e:
            transcript.append({"turn": turn, "error": f"generate() failed: {e}"})
            break
        transcript.append({"turn": turn, "model": reply})
        messages.append({"role": "assistant", "content": reply})

        call = parse_tool_call(reply)
        if call is None:
            messages.append({"role": "user", "content": (
                "No valid tool_call block found. Respond with exactly one "
                "```tool_call\\n{\"tool\": ..., \"args\": {...}}\\n``` block."
            )})
            continue

        result = dispatch(sandbox, call)
        transcript.append({"turn": turn, "tool_call": call, "result": result})
        messages.append({"role": "user", "content": f"Tool result:\n{json.dumps(result)[:MAX_STDOUT_CHARS]}"})

        if call.get("tool") == "submit_patch":
            break

    if not sandbox.patch_submitted:
        sandbox.submit_patch()

    (LOGS_DIR / f"{instance_id}.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    (PATCHES_DIR / f"{instance_id}.patch").write_text(sandbox.submitted_patch, encoding="utf-8")

    verification = verify_task(instance_id, sandbox.submitted_patch, task, scratch)
    return {
        "instance_id": instance_id, "repo": task["repo"],
        "turns_used": turn, "tool_calls_used": sandbox.tool_calls_used,
        "patch_size": len(sandbox.submitted_patch),
        **verification,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
results = []
for task in sample:
    scratch = Path(tempfile.mkdtemp(prefix=f"{task['instance_id']}_"))
    try:
        res = run_agent_on_task(task, scratch)
    except Exception as e:
        res = {"instance_id": task["instance_id"], "repo": task["repo"], "resolved": False,
                "reason": f"exception: {e}\n{traceback.format_exc()[-2000:]}"}
    results.append(res)
    log(f"  -> resolved={res.get('resolved')} reason={res.get('reason')}")
    shutil.rmtree(scratch, ignore_errors=True)

resolved_count = sum(1 for r in results if r.get("resolved"))
summary = {
    "note": "APPROXIMATE local harness -- not the real grader. Same harness + same 10-task sample as exp3 (zero-shot baseline) for apples-to-apples comparison. See THINKING_AND_DECISIONS.md section 3.",
    "model": "gemma-4-e4b-it (4-bit NF4) + exp4 LoRA adapter",
    "n_tasks": len(results),
    "resolved": resolved_count,
    "resolution_rate": resolved_count / len(results) if results else 0.0,
    "results": results,
}
(RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
log(f"DONE. resolution_rate={summary['resolution_rate']:.3f} ({resolved_count}/{len(results)})")
