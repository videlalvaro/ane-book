#!/usr/bin/env python3
"""Quick parity check: old T=1 shard vs new RangeDim shard (T=1) for [0,5) layers.
Same weights → cosine similarity should be ≥ 0.9999.
"""
import coremltools as ct
import numpy as np
import sys

OLD  = "models/hymt/ane/hymt_q8_s0_5.mlpackage"
NEW  = "models/hymt/ane/hymt_rangedim_layer0_5_q8.mlpackage"

D = 2048
MAX_SEQ = 512
T = 1            # Both shards support T=1; RangeDim supports 1..4

np.random.seed(42)
hidden = np.random.randn(1, D, T, 1).astype(np.float16)
cos_vals = np.random.randn(64).astype(np.float16)   # base 64 rope values
attn_mask   = np.zeros((1, 1, T, MAX_SEQ), dtype=np.float16)
kv_write    = np.zeros((1, 1, MAX_SEQ, T), dtype=np.float16)
kv_write[0, 0, 0, 0] = 1.0   # write position 0

def run(path):
    model  = ct.models.MLModel(path)
    inames = set(model._model_input_names_set)
    hidden_key  = "hidden"  if "hidden"   in inames else "x"
    print(f"  inputs: {sorted(inames)}")

    # Detect rope_cos rank from model spec
    spec = model.get_spec()
    cos_inp = next(i for i in spec.description.input if "cos" in i.name)
    cos_shape = list(cos_inp.type.multiArrayType.shape)
    print(f"  cos_shape in spec: {cos_shape}")

    if len(cos_shape) == 2:
        # Old format: [1, 64] per token
        rc = cos_vals.reshape(1, 64)
        rs = cos_vals.reshape(1, 64)  # use same vals for sin (parity test)
        cos_key, sin_key = "rope_cos", "rope_sin"
    else:
        # New format: [1, 64, T, 1]
        rc = cos_vals.reshape(1, 64, 1, 1)
        rs = cos_vals.reshape(1, 64, 1, 1)
        cos_key = "cos" if "cos" in inames else "rope_cos"
        sin_key = "sin" if "sin" in inames else "rope_sin"

    inputs = {
        hidden_key:   hidden,
        cos_key:      rc,
        sin_key:      rs,
        "attn_mask":  attn_mask,
        "kv_write_mask": kv_write,
    }
    out = model.predict(inputs)
    okey = [k for k in out if "hidden" in k or "output" in k]
    if not okey:
        okey = list(out.keys())
    arr = out[okey[0]]
    return arr.flatten().astype(np.float32)

print(f"Loading OLD  {OLD} …")
old_out = run(OLD)
print(f"Loading NEW  {NEW} …")
new_out = run(NEW)

cos_sim = np.dot(old_out, new_out) / (np.linalg.norm(old_out) * np.linalg.norm(new_out) + 1e-12)
l2_rel  = np.linalg.norm(old_out - new_out) / (np.linalg.norm(old_out) + 1e-12)

print(f"\nCosine similarity : {cos_sim:.6f}  (threshold ≥ 0.9999)")
print(f"Relative L2       : {l2_rel:.6f}")

if cos_sim >= 0.9999:
    print("PASS ✓")
else:
    print("FAIL ✗")
    sys.exit(1)
