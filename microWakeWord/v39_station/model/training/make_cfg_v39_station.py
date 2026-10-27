#!/usr/bin/env python3
"""cfg_v39_station.yaml = v38/cfg_v38.yaml + two station blocks (mirror make_cfg_v38.py), by text
substitution so everything else stays byte-identical:
  features_station_pos  (18 / 1 / truth / truncate_start) — screened station «захар» word-crops;
  features_station_neg  (14 / 10 / false / random)        — screened station room hard-negatives;
train_dir -> v39_station/models/zakhar_v39_station (export reads weights from train_dir)."""
V38 = "/home/claude/zakhar-mww/v38"
V   = "/home/claude/zakhar-mww/v39_station"
s = open(f"{V38}/cfg_v38.yaml").read()
td = "train_dir: /home/claude/zakhar-mww/v38/models/zakhar_v38\n"
anchor = "training_steps:\n"
new = (f"- features_dir: {V}/features/features_station_pos\n  sampling_weight: 18.0\n  penalty_weight: 1.0\n"
       "  truth: true\n  truncation_strategy: truncate_start\n  type: mmap\n"
       f"- features_dir: {V}/features/features_station_neg\n  sampling_weight: 14.0\n  penalty_weight: 10.0\n"
       "  truth: false\n  truncation_strategy: random\n  type: mmap\n")
assert s.count(td) == 1, "train_dir anchor not unique"
assert s.count(anchor) == 1, "training_steps anchor not unique"
out = s.replace(td, f"train_dir: {V}/models/zakhar_v39_station\n").replace(anchor, new + anchor)
open(f"{V}/cfg_v39_station.yaml", "w").write(out)
import yaml
c = yaml.safe_load(open(f"{V}/cfg_v39_station.yaml"))
print("cfg_v39_station.yaml", len(c["features"]), "blocks, train_dir", c["train_dir"])
assert c["features"][-2]["features_dir"].endswith("features_station_pos") and c["features"][-2]["truth"] is True
assert c["features"][-1]["features_dir"].endswith("features_station_neg") and c["features"][-1]["truth"] is False
print("OK: last two blocks = station_pos(truth True w18) + station_neg(truth False w14)")
