#!/usr/bin/env python3
"""cfg_v37.yaml = v36/cfg_v36.yaml + the owner's two read-session blocks, by text substitution so everything
else stays byte-identical:
  features_owner_pos  the block settings of features_studio2_pos (18 / 1 / truth / truncate_start);
  features_owner_neg  the block settings of features_confus_fused_neg (8 / 8 / false / truncate_start);
train_dir -> v37/models/zakhar_v37 (the export reads the weights from train_dir)."""
V36, V = "/home/claude/zakhar-mww/v36", "/home/claude/zakhar-mww/v37"
s = open(f"{V36}/cfg_v36.yaml").read()
td = f"train_dir: {V36}/models/zakhar_v36\n"
last = (f"- features_dir: {V36}/features/features_confus_fused_neg\n  sampling_weight: 8.0\n  penalty_weight: 8.0\n"
        "  truth: false\n  truncation_strategy: truncate_start\n  type: mmap\n")
new = (f"- features_dir: {V}/features/features_owner_pos\n  sampling_weight: 18.0\n  penalty_weight: 1.0\n"
       "  truth: true\n  truncation_strategy: truncate_start\n  type: mmap\n"
       f"- features_dir: {V}/features/features_owner_neg\n  sampling_weight: 8.0\n  penalty_weight: 8.0\n"
       "  truth: false\n  truncation_strategy: truncate_start\n  type: mmap\n")
assert s.count(td) == 1 and s.count(last) == 1
open(f"{V}/cfg_v37.yaml", "w").write(s.replace(td, f"train_dir: {V}/models/zakhar_v37\n").replace(last, last + new))
import yaml
c = yaml.safe_load(open(f"{V}/cfg_v37.yaml"))
print("cfg_v37.yaml", len(c["features"]), "blocks, train_dir", c["train_dir"])
