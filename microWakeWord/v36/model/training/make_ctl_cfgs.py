#!/usr/bin/env python3
"""Configs of the two control runs, by text substitution so everything else stays byte-identical:
  cfg_ctl_v35node.yaml = cfg_v35_node.yaml with train_dir -> models/ctl_v35node (the v35 recipe on the rebuilt
                         blocks, no new data: measures the noise of the rebuild);
  cfg_v36_nobg.yaml    = cfg_v36.yaml without the studio background block, train_dir -> models/v36_nobg.
The export reads the weights from train_dir, so train_dir must be the model's own dir."""
W = "/home/claude/zakhar-mww/v36"
s = open(f"{W}/cfg_v35_node.yaml").read()
td = "train_dir: /tmp/zstudio/v35/models/zakhar_v35\n"
assert s.count(td) == 1
open(f"{W}/cfg_ctl_v35node.yaml", "w").write(s.replace(td, f"train_dir: {W}/models/ctl_v35node\n"))

s = open(f"{W}/cfg_v36.yaml").read()
td = f"train_dir: {W}/models/zakhar_v36\n"
bg = (f"- features_dir: {W}/features/features_studio_bg_neg\n  sampling_weight: 14.0\n  penalty_weight: 10.0\n"
      "  truth: false\n  truncation_strategy: random\n  type: mmap\n")
assert s.count(td) == 1 and s.count(bg) == 1
open(f"{W}/cfg_v36_nobg.yaml", "w").write(s.replace(td, f"train_dir: {W}/models/v36_nobg\n").replace(bg, ""))

import yaml
for n in ("cfg_ctl_v35node.yaml", "cfg_v36_nobg.yaml"):
    c = yaml.safe_load(open(f"{W}/{n}"))
    print(n, len(c["features"]), "blocks, train_dir", c["train_dir"])
