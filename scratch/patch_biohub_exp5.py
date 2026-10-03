import json

path = "competitions/biohub_cell_tracking/exp5_division_precision_sota/harmonic_division_sota.ipynb"
nb = json.load(open(path, "r", encoding="utf-8"))
src = "".join(nb["cells"][2]["source"])

src = src.replace('os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"', 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.15"')
src = src.replace('os.environ["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "2.25"', 'os.environ["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "1.75"')
src = src.replace('os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "9.0"', 'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "10.0"')

nb["cells"][2]["source"] = [src]
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
print("Updated Cell 2 in harmonic_division_sota.ipynb successfully.")
