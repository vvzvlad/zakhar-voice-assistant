#!/usr/bin/env python3
"""cfg_v35_node.yaml = recipe/cfg_v35.yaml with the five lost /tmp feature dirs pointed at their rebuilds;
cfg_v36.yaml = cfg_v35_node.yaml + train_dir v36 + the new blocks given on the command line.
Text substitution on the recovered file, so everything else stays byte-identical.
Usage: make_cfgs.py <new_block_spec> ...   spec = features_dir:sampling_weight:penalty_weight:truth:truncation"""
import sys
W = "/home/claude/zakhar-mww/v36"
SP = "/tmp/claude-1000/-home-claude/71fbe714-4eba-4120-a077-75d50a00bed2/scratchpad"
RELOC = {f"{SP}/t8/run1/features_real_pos": f"{W}/features/features_real_pos",
         f"{SP}/t8/run2/features_hardneg_speech": f"{W}/features/features_hardneg_speech",
         f"{SP}/t8/run3/features_panel_pos": f"{W}/features/features_panel_pos",
         "/tmp/zstudio/v34/features_studio_pos": f"{W}/features/features_studio_pos",
         "/tmp/zstudio/v35/features_studio2_pos": f"{W}/features/features_studio2_pos"}
s = open(f"{W}/recipe/cfg_v35.yaml").read()
for a, b in RELOC.items():
    assert s.count(f"- features_dir: {a}\n") == 1, a
    s = s.replace(f"- features_dir: {a}\n", f"- features_dir: {b}\n")
assert "/tmp/" not in s.split("train_dir:")[1].split("\n", 1)[1], "a /tmp feature path is left"
open(f"{W}/cfg_v35_node.yaml", "w").write(s)

old_td = "train_dir: /tmp/zstudio/v35/models/zakhar_v35\n"
assert s.count(old_td) == 1
s = s.replace(old_td, f"train_dir: {W}/models/zakhar_v36\n")
anchor = "training_steps:\n"
blocks = ""
for spec in sys.argv[1:]:
    d, sw, pw, truth, tr = spec.split(":")
    blocks += (f"- features_dir: {d}\n  sampling_weight: {float(sw)}\n  penalty_weight: {float(pw)}\n"
               f"  truth: {truth}\n  truncation_strategy: {tr}\n  type: mmap\n")
assert s.count(anchor) == 1
s = s.replace(anchor, blocks + anchor)
open(f"{W}/cfg_v36.yaml", "w").write(s)
import yaml
c = yaml.safe_load(s)
print(f"cfg_v36: {len(c['features'])} blocks, train_dir {c['train_dir']}")
for b in c["features"][-len(sys.argv[1:]):] if sys.argv[1:] else []:
    print("  new:", b)
