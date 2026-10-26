"""Print the ppo.py arguments that reproduce a saved run (all recorded settings, except seed and name).

    python rerun_cmd.py results/runs/L_distill_seed1.json
"""
import json
import sys

cfg = json.load(open(sys.argv[1]))["config"]
args = []
for k, v in cfg.items():
    if k in ("seed", "name") or v is None or v is False:
        continue
    flag = "--" + k.replace("_", "-")
    args += [flag] if v is True else [flag, str(v)]
print(" ".join(args))
