import json
from pathlib import Path

nb_path = Path("competitions/ieee_traffic_flow/exp10_hydrodynamic_delayed_onset/notebook.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c16 = "".join(nb["cells"][16]["source"])

target1 = 'dyn_bottlenecks = set(at_t0.sort_values("_margin").head(3)["link_id"].astype(str))'
replace1 = 'dyn_bottlenecks = set(at_t0[at_t0["_margin"] <= 0.85].sort_values("_margin").head(3)["link_id"].astype(str))'

target2 = "if step_num >= 1 and lid in dyn_bottlenecks:"
replace2 = "if step_num >= 4 and lid in dyn_bottlenecks:  # Strict T>=25m delayed onset threshold"

assert target1 in c16, "target1 not found"
assert target2 in c16, "target2 not found"

c16 = c16.replace(target1, replace1).replace(target2, replace2)
nb["cells"][16]["source"] = [c16]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("SUCCESS: Patched Cell 16 for Exp 10 Hydrodynamic Delayed Onset!")
