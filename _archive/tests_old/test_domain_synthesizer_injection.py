"""Behavioral test for the domain synthesizer code-injection fix.

Root cause: bio_synthesizer.py, dicom_synthesizer.py, and
materials_synthesizer.py generate standalone Python scripts by directly
interpolating caller-supplied strings into quoted literals (sequence
wrapped in triple quotes, dicom_dir in a raw double-quoted string, etc.)
with no escaping. Any input containing the matching quote sequence
breaks out of the literal in the GENERATED script — turning arbitrary text
into arbitrary Python source that then gets written to disk and
potentially executed via run_command. Currently unreachable (these
synthesizers have no callers anywhere in the codebase), but that's exactly
why this needed fixing now rather than after something wires them in.

This compiles the generated scripts with adversarial input designed to
break out of a naive quoted-literal interpolation, and separately verifies
the assignment round-trips to the exact original value (not just "doesn't
crash") - proving the fix is actually correct, not merely non-crashing.
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  - {detail}" if detail and not cond else ""))


ADVERSARIAL_STRINGS = [
    '"""; import os; os.system("calc"); x = """',   # triple-quote breakout attempt
    'nr"; import sys; sys.exit(1); db = "nr',        # double-quote breakout attempt
    "line1\nline2\nwith \"quotes\" and 'apostrophes'",
    "C:\\Users\\test\\path\\with\\backslashes",
    "",  # empty string edge case
]


def assert_compiles_and_roundtrips(script: str, var_name: str, expected_value, label: str):
    try:
        compile(script, "<generated>", "exec")
    except SyntaxError as e:
        check(f"{label}: generated script is valid Python", False, f"SyntaxError: {e}")
        return
    check(f"{label}: generated script is valid Python", True)

    # Extract just the target assignment line(s) and exec them in isolation
    # (the full script needs Bio/pydicom/numpy, which aren't installed here
    # and aren't the point of this test - we're verifying the literal, not
    # the science).
    match = re.search(rf"^{re.escape(var_name)} = (.+)$", script, re.MULTILINE)
    if not match:
        check(f"{label}: assignment line found in output", False, "no match")
        return
    ns = {}
    try:
        exec(f"{var_name} = {match.group(1)}", ns)
    except SyntaxError as e:
        check(f"{label}: assignment line itself is valid Python", False, str(e))
        return
    check(f"{label}: round-trips to the exact original value",
          ns[var_name] == expected_value, f"got {ns[var_name]!r}, expected {expected_value!r}")


from autobot.agent.domain.bio_synthesizer import BioInformaticsSynthesizer
from autobot.agent.domain.dicom_synthesizer import DICOMVolumeSynthesizer
from autobot.agent.domain.materials_synthesizer import MaterialsScienceSynthesizer

for adv in ADVERSARIAL_STRINGS:
    script = BioInformaticsSynthesizer.generate_blast_script(adv, program=adv, database=adv, output_csv=adv)
    assert_compiles_and_roundtrips(script, "sequence", adv, f"bio[{adv[:20]!r}]")

    script = DICOMVolumeSynthesizer.generate_volume_script(adv, hu_min_threshold=300, output_json=adv)
    assert_compiles_and_roundtrips(script, "dicom_dir", adv, f"dicom[{adv[:20]!r}]")

    script = MaterialsScienceSynthesizer.generate_cif_script(adv, output_path=adv)
    assert_compiles_and_roundtrips(script, "output_path", adv, f"materials[{adv[:20]!r}]")

    script = MaterialsScienceSynthesizer.generate_vesta_launch_script(adv, vesta_exe=adv)
    assert_compiles_and_roundtrips(script, "cif_path", adv, f"vesta[{adv[:20]!r}]")

# formula gets sanitized (not round-tripped exactly) for its nested-string
# use, so it needs its own check: valid chars survive, dangerous ones don't
# break the generated script.
tricky_formula = 'Fe2O3"""; import os; os.system("calc") #'
script = MaterialsScienceSynthesizer.generate_cif_script(tricky_formula, output_path="out.cif")
try:
    compile(script, "<generated>", "exec")
    check("materials: adversarial formula still compiles", True)
except SyntaxError as e:
    check("materials: adversarial formula still compiles", False, str(e))
check("materials: formula's Python-literal use (repr'd) preserves the original exactly",
      f"formula = {tricky_formula!r}" in script)
check("materials: formula's nested-CIF use is sanitized to safe characters only",
      f'data_{tricky_formula}' not in script and "data_Fe2O3" in script)

# hu_min_threshold: verify non-numeric input is rejected rather than injected
try:
    DICOMVolumeSynthesizer.generate_volume_script("dir", hu_min_threshold="300; os.system('x')")
    check("dicom: non-numeric threshold is rejected, not injected", False, "no exception raised")
except (ValueError, TypeError):
    check("dicom: non-numeric threshold is rejected, not injected", True)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
