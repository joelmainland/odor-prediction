#!/usr/bin/env python3
"""Export the OpenPOM checkpoint as a flat binary + JSON manifest for in-browser
inference (docs/pom-worker.js).

Two simplifications are applied here so the browser doesn't have to:
  - Optimizer state is dropped.
  - The FFN's batch-norm layers are folded into the preceding Linear, since at
    inference time BN is just an affine map. bn2 is kept separate because the POM
    embedding is read off *before* it.

Precision: fp32 (default) reproduces the reference openpom/DGL stack to ~2e-6, so the
browser and a local openpom run give the same answer. fp16 halves the download
(3.9 MB vs 7.9 MB gzipped) but deviates by ~1.4e-3, which reorders the displayed
top-8 descriptors for ~3% of molecules — measured, not estimated; see
scripts/verify_pom_js.mjs.
"""
import argparse, json, os
import numpy as np
import torch

DEFAULT_CK = os.path.expanduser(
    "~/Monell Dropbox/Mainland Lab Team Folder/Projects/OpenPOM/code/pom_checkpoint.pt")

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=DEFAULT_CK)
ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
ap.add_argument("--out-bin", default="docs/data/pom_weights.bin")
ap.add_argument("--out-manifest", default="docs/data/pom_weights.json")
args = ap.parse_args()

sd = {k: v.numpy().astype(np.float64)
      for k, v in torch.load(args.checkpoint, map_location="cpu")["model_state_dict"].items()}

EPS = 1e-5


def bn_affine(prefix):
    """BatchNorm1d in eval mode as (scale, shift)."""
    scale = sd[f"{prefix}.weight"] / np.sqrt(sd[f"{prefix}.running_var"] + EPS)
    return scale, sd[f"{prefix}.bias"] - sd[f"{prefix}.running_mean"] * scale


def fold(linear, bn):
    """Fold a following BatchNorm into a Linear: y = BN(xW' + b)."""
    scale, shift = bn_affine(bn)
    return (sd[f"{linear}.weight"] * scale[:, None],
            sd[f"{linear}.bias"] * scale + shift)


w0, b0 = fold("ffn.linears.0", "ffn.batchnorms.0")
w1, b1 = fold("ffn.linears.1", "ffn.batchnorms.1")
bn2_scale, bn2_shift = bn_affine("ffn.batchnorms.2")

tensors = {
    "proj_node.w": sd["mpnn.project_node_feats.0.weight"],
    "proj_node.b": sd["mpnn.project_node_feats.0.bias"],
    "edge_func.0.w": sd["mpnn.gnn_layer.edge_func.0.weight"],
    "edge_func.0.b": sd["mpnn.gnn_layer.edge_func.0.bias"],
    "edge_func.2.w": sd["mpnn.gnn_layer.edge_func.2.weight"],
    "edge_func.2.b": sd["mpnn.gnn_layer.edge_func.2.bias"],
    "conv.bias": sd["mpnn.gnn_layer.bias"],
    "gru.w_ih": sd["mpnn.gru.weight_ih_l0"],
    "gru.w_hh": sd["mpnn.gru.weight_hh_l0"],
    "gru.b_ih": sd["mpnn.gru.bias_ih_l0"],
    "gru.b_hh": sd["mpnn.gru.bias_hh_l0"],
    "proj_edge.w": sd["project_edge_feats.0.weight"],
    "proj_edge.b": sd["project_edge_feats.0.bias"],
    "ffn.0.w": w0, "ffn.0.b": b0,
    "ffn.1.w": w1, "ffn.1.b": b1,
    "ffn.2.w": sd["ffn.linears.2.weight"], "ffn.2.b": sd["ffn.linears.2.bias"],
    "bn2.scale": bn2_scale, "bn2.shift": bn2_shift,
    "ffn.3.w": sd["ffn.linears.3.weight"], "ffn.3.b": sd["ffn.linears.3.bias"],
}
for L in (0, 1):
    for part in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        tensors[f"lstm.{L}.{part}"] = sd[f"readout_set2set.lstm.{part}_l{L}"]

dtype = np.float16 if args.precision == "fp16" else np.float32
blob, manifest, offset = bytearray(), [], 0
for name, arr in tensors.items():
    raw = np.ascontiguousarray(arr, dtype=dtype).tobytes()
    manifest.append({"name": name, "shape": list(arr.shape), "offset": offset})
    blob += raw
    offset += len(raw)

with open(args.out_bin, "wb") as f:
    f.write(blob)
with open(args.out_manifest, "w") as f:
    json.dump({"dtype": args.precision, "bytes": len(blob), "tensors": manifest}, f,
              separators=(",", ":"))
    f.write("\n")

n = sum(a.size for a in tensors.values())
print(f"wrote {len(manifest)} tensors, {n:,} params, {len(blob)/1e6:.2f} MB "
      f"({args.precision}) -> {args.out_bin}")
