#!/usr/bin/env python3
"""cfg_v38.yaml = v37/cfg_v37.yaml + the two blocks of the speaker's rejected/empty wakes from the run log, by
text substitution so everything else stays byte-identical:
  features_runs_pos  the block settings of features_studio2_pos (18 / 1 / truth / truncate_start);
  features_runs_neg  the block settings of features_studio_bg_neg (14 / 10 / false / random);
train_dir -> v38/models/zakhar_v38 (the export reads the weights from train_dir)."""
V37, V = "/home/claude/zakhar-mww/v37", "/home/claude/zakhar-mww/v38"
s = open(f"{V37}/cfg_v37.yaml").read()
td = f"train_dir: {V37}/models/zakhar_v37\n"
last = (f"- features_dir: {V37}/features/features_owner_neg\n  sampling_weight: 8.0\n  penalty_weight: 8.0\n"
        "  truth: false\n  truncation_strategy: truncate_start\n  type: mmap\n")
new = (f"- features_dir: {V}/features/features_runs_pos\n  sampling_weight: 18.0\n  penalty_weight: 1.0\n"
       "  truth: true\n  truncation_strategy: truncate_start\n  type: mmap\n"
       f"- features_dir: {V}/features/features_runs_neg\n  sampling_weight: 14.0\n  penalty_weight: 10.0\n"
       "  truth: false\n  truncation_strategy: random\n  type: mmap\n")
assert s.count(td) == 1 and s.count(last) == 1
open(f"{V}/cfg_v38.yaml", "w").write(s.replace(td, f"train_dir: {V}/models/zakhar_v38\n").replace(last, last + new))
import yaml
c = yaml.safe_load(open(f"{V}/cfg_v38.yaml"))
print("cfg_v38.yaml", len(c["features"]), "blocks, train_dir", c["train_dir"])
