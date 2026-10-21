#!/usr/bin/env python3
"""Compare two streaming tflite models: op multiset, input/output tensors (name, shape, dtype,
quantization), and the forbidden op strings (dequant/rsqrt/square). Prints SAME/DIFF per item.
Usage: compare_tflite.py <reference.tflite> <candidate.tflite>"""
import sys, re, collections
import tensorflow as tf


def describe(path):
    # no default XNNPACK delegate: it would fold ops into DELEGATE nodes and hide them from the list
    it = tf.lite.Interpreter(model_path=path, experimental_op_resolver_type=
                             tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ops = collections.Counter(o["op_name"] for o in it._get_ops_details())
    io = lambda ds: [(d["name"], tuple(int(x) for x in d["shape"]), d["dtype"].__name__,
                      tuple(round(float(q), 8) for q in d["quantization"])) for d in ds]
    blob = open(path, "rb").read().decode("latin-1")
    bad = len(re.findall(r"dequant|rsqrt|square", blob, flags=re.I))
    return {"ops": dict(sorted(ops.items())), "inputs": io(it.get_input_details()),
            "outputs": io(it.get_output_details()), "forbidden_strings": bad}


ref, cand = describe(sys.argv[1]), describe(sys.argv[2])
for k in ("ops", "inputs", "outputs"):
    print(f"{k:8s} {'SAME' if ref[k] == cand[k] else 'DIFF'}")
    print(f"   ref : {ref[k]}")
    if ref[k] != cand[k]: print(f"   cand: {cand[k]}")
print(f"forbidden dequant/rsqrt/square strings: ref {ref['forbidden_strings']}, cand {cand['forbidden_strings']}")
ok = all(ref[k] == cand[k] for k in ("ops", "inputs", "outputs")) and cand["forbidden_strings"] == 0
print("COMPARE_OK" if ok else "COMPARE_FAIL")
