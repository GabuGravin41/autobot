"""Behavioral test for autobot/diagnostics.py (the `autobot --doctor` command).

Runs on pure stdlib so it works even when Autobot's own dependencies are
missing - which is exactly the situation the doctor exists to diagnose.
"""
import importlib.util
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "autobot" / "diagnostics.py"
spec = importlib.util.spec_from_file_location("diagnostics", SRC)
d = importlib.util.module_from_spec(spec)
sys.modules["diagnostics"] = d
spec.loader.exec_module(d)

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  - {detail}" if detail and not cond else ""))


# ---- runs to completion without raising, whatever the environment ----
checks = d.run_all()
check("run_all returns checks", isinstance(checks, list) and len(checks) > 5, str(len(checks)))
check("every check has a valid status",
      all(c.status in (d.OK, d.WARN, d.FAIL) for c in checks))
check("every check has a name", all(c.name for c in checks))

# ---- failures must be actionable: a FAIL with no suggested fix is useless ----
unhelpful = [c.name for c in checks if c.status == d.FAIL and not c.fix]
check("all FAIL checks include a fix", not unhelpful, f"missing fix: {unhelpful}")

# ---- output must survive a Windows cp1252 console ----
blob = "".join(f"{c.name}{c.detail}{c.fix}" for c in checks)
non_ascii = sorted({ch for ch in blob if ord(ch) > 127})
check("output is pure ASCII (cp1252-safe)", not non_ascii, f"found {non_ascii}")
try:
    blob.encode("cp1252")
    check("output encodes to cp1252", True)
except UnicodeEncodeError as e:
    check("output encodes to cp1252", False, str(e))

# ---- never leaks secret values ----
SENTINEL = "sk-secret-value-do-not-print-12345"
os.environ["OPENROUTER_API_KEY"] = SENTINEL
try:
    llm_checks = d.check_llm_config()
    rendered = " ".join(f"{c.name}{c.detail}{c.fix}" for c in llm_checks)
    check("API key value is never printed", SENTINEL not in rendered, rendered)
    check("but key presence IS reported",
          any(c.status == d.OK and "set:" in c.detail for c in llm_checks), rendered)
finally:
    del os.environ["OPENROUTER_API_KEY"]

# ---- missing key is reported as a blocking failure ----
saved = {k: os.environ.pop(k) for k in
         ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
          "GOOGLE_API_KEY", "ANTHROPIC_API_KEY") if k in os.environ}
try:
    check("no key -> FAIL", any(c.status == d.FAIL for c in d.check_llm_config()))
finally:
    os.environ.update(saved)

# ---- approval mode validation ----
os.environ["AUTOBOT_APPROVAL_MODE"] = "nonsense"
try:
    check("invalid approval mode -> FAIL", d.check_approval_mode().status == d.FAIL)
finally:
    os.environ["AUTOBOT_APPROVAL_MODE"] = "trusted"
    check("trusted mode still notes IRREVERSIBLE gating",
          "IRREVERSIBLE" in d.check_approval_mode().detail,
          d.check_approval_mode().detail)
    del os.environ["AUTOBOT_APPROVAL_MODE"]

# ---- ANTHROPIC_API_KEY set but the 'anthropic' package missing -> FAIL,
# not a silent "OK" that only breaks at the first real run ----
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-value"
try:
    anth_checks = d.check_llm_config()
    pkg_check = next((c for c in anth_checks if c.name == "package: anthropic"), None)
    check("anthropic key without package triggers a check", pkg_check is not None)
    # This environment has no network access to install `anthropic`, so this
    # assertion is itself proof the check fires correctly when the package
    # really is missing - not a mock.
    if pkg_check is not None:
        if d._module_present("anthropic"):
            check("installed anthropic package reported as OK", pkg_check.status == d.OK)
        else:
            check("missing anthropic package reported as FAIL, not silent OK",
                  pkg_check.status == d.FAIL, pkg_check.detail)
            check("that FAIL still has an actionable fix", bool(pkg_check.fix))
finally:
    del os.environ["ANTHROPIC_API_KEY"]

# ---- exit code contract: nonzero only when something is blocking ----
code = d.report(checks)
has_fail = any(c.status == d.FAIL for c in checks)
check("exit code matches blocking state", (code != 0) == has_fail, f"code={code} has_fail={has_fail}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    if __name__ == '__main__':
        sys.exit(1)
