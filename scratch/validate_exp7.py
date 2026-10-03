import os, sys, yaml, zipfile, shutil
from pathlib import Path

bundle = Path("competitions/gemma_4_developer_agent/exp7_sequential_pipeline/agent_bundle")
print("Validating Exp 7 bundle at:", bundle.resolve())

class SafeIncludeLoader(yaml.SafeLoader):
    pass
SafeIncludeLoader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))

# 1. Root agent
root_yaml = bundle / "agent.yaml"
assert root_yaml.exists(), "agent.yaml missing"
with open(root_yaml, "r", encoding="utf-8") as f:
    root_data = yaml.load(f, Loader=SafeIncludeLoader)

print("Root class:", root_data["agent_class"])
print("Sub-agents count:", len(root_data["sub_agents"]))
assert root_data["agent_class"] == "SequentialAgent"

# 2. Check each sub-agent
models = set()
for sub in root_data["sub_agents"]:
    cpath = sub["config_path"]
    sub_path = (root_yaml.parent / cpath).resolve()
    assert sub_path.exists(), f"Missing sub-agent: {cpath}"
    with open(sub_path, "r", encoding="utf-8") as f:
        sub_data = yaml.load(f, Loader=SafeIncludeLoader)
    models.add(sub_data["model"])
    print(f"  - Sub-agent {sub_data['name']}: model={sub_data['model']}, tools={sub_data['tools']}")

print("Declared models across pipeline:", models)
assert models == {"gemma-4-31b-it-qat-w4a16-ct"}, "Must declare single base model"

# 3. Test packaging
zip_out = Path("competitions/gemma_4_developer_agent/exp7_sequential_pipeline/submission.zip")
zip_out.unlink(missing_ok=True)

with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(p for p in bundle.rglob("*") if p.is_file()):
        rel_path = str(p.relative_to(bundle)).replace("\\", "/")
        info = zipfile.ZipInfo(rel_path, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, p.read_bytes())

print(f"Package size: {zip_out.stat().st_size} bytes")
with zipfile.ZipFile(zip_out, "r") as zf:
    print("Zip files:")
    for name in zf.namelist():
        print(f"  {name}")

print("\n✅ EXP 7 BUNDLE VALIDATION PASSED!")
