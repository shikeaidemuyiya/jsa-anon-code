# -*- coding: utf-8 -*-
"""Round-2 review accounting script.

Produces numbers for:
  A. MobileNetV2 exact FLOPs ratios (baseline vs stem-decomposed k=7/9/13)
  B. 8-bit index FCQ variant: accuracy + bits/weight (CIFAR ResNet-18, IN-100 accounted analytically)
  C. Compression-stage wall-clock cost table (fuse, decompose, FCQ) + peak alloc
  D. ResNet-18 actual rank k ranges per tau
  E. Two-stage footprint exceeding baseline: concrete counterexample bytes
  F. Table 3 exact byte values in KiB
Writes round2_fixes.json.
"""
import copy, json, os, time, tracemalloc
import numpy as np
import torch
import torch.nn as nn

from compress import (fuse_bn, decompose_convs, collect_ks, choose_K_adaptive,
                      cluster_quantize, QuantizedFC, diff_encode_bias)
from run_experiments import quantize_all_fc, compressed_size
from supplement import count_flops
from baselines import get_loaders
from finetune_utils import evaluate
from train_baseline import resnet18_cifar, mobilenetv2_cifar

OUT = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = {}

# ---------------- A. MobileNetV2 exact FLOPs ----------------
mb = mobilenetv2_cifar()
mb.load_state_dict(torch.load(os.path.join(OUT, "mobilenetv2_best.pth"), map_location="cpu", weights_only=True))
fused = fuse_bn(copy.deepcopy(mb))
f0 = count_flops(fused.cpu(), in_shape=(1, 3, 32, 32))
res["mobilenetv2"] = {"baseline_flops": f0}
for tau in [0.7, 0.8, 0.9]:
    d = decompose_convs(fuse_bn(copy.deepcopy(mb)), tau)
    f1 = count_flops(d.cpu(), in_shape=(1, 3, 32, 32))
    res["mobilenetv2"][f"tau{tau}"] = {"flops": f1, "reduction": f0 / f1,
                                       "ks": collect_ks(d)}
print("A mobilenetv2:", json.dumps(res["mobilenetv2"], default=float), flush=True)

# ---------------- B. 8-bit index FCQ (CIFAR ResNet-18, tau=0.9 full pipeline not needed; use tau=0.7 student ckpt)
tl, vl, in_size, nc = get_loaders("cifar10", workers=0)
from compress import fuse_bn as _fb
r18 = resnet18_cifar()
r18.load_state_dict(torch.load(os.path.join(OUT, "resnet18_best.pth"), map_location="cpu", weights_only=True))
m = decompose_convs(_fb(copy.deepcopy(r18)), 0.7)
m = m.to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=5e-4)
crit = nn.CrossEntropyLoss()
for ep in range(3):
    m.train()
    for x, y in tl:
        x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        opt.zero_grad(set_to_none=True); crit(m(x), y).backward(); opt.step()
m.eval().cpu()

class QuantizedFC8(nn.Module):
    """uint8 index variant (valid for K<=256)."""
    def __init__(self, fc, centers, idx):
        super().__init__()
        self.out_f, self.in_f = fc.weight.shape
        self.register_buffer("codebook", torch.from_numpy(centers))
        self.register_buffer("idx", torch.from_numpy(idx.astype(np.uint8)))
        if fc.bias is not None:
            self.register_buffer("bias", fc.bias.data.clone())
        else:
            self.bias = None
    def forward(self, x):
        w = self.codebook[self.idx.long()]
        return torch.nn.functional.linear(x, w.view(self.out_f, self.in_f), self.bias)

def build_fcq(model, idx_dtype_variant):
    info = []
    out = copy.deepcopy(model)
    def _recurse(container):
        for name, mm in list(container.named_children()):
            if isinstance(mm, nn.Linear):
                sel = choose_K_adaptive(mm.weight.data.numpy(), cache_bytes=32*1024)
                K = sel["K"] if sel else 64
                centers, idx, counts = cluster_quantize(mm.weight.data.numpy(), K)
                cls = QuantizedFC8 if idx_dtype_variant == "u8" else QuantizedFC
                q = cls(mm, centers, idx)
                setattr(container, name, q)
                info.append({"K": K, "n_weights": mm.weight.numel(),
                             "idx_bytes": idx.size * (1 if idx_dtype_variant == "u8" else 2),
                             "codebook_bytes": centers.size * 4})
            else:
                _recurse(mm)
    _recurse(out)
    return out, info

for variant in ["u16", "u8"]:
    mq, info = build_fcq(m, variant)
    acc = evaluate(mq, vl, DEV)
    idx_b = sum(r["idx_bytes"] for r in info); cb_b = sum(r["codebook_bytes"] for r in info)
    n_w = sum(r["n_weights"] for r in info)
    bits = (idx_b + cb_b) * 8 / n_w
    res[f"fcq_{variant}"] = {"acc": acc, "Ks": [r["K"] for r in info],
                             "idx_bytes": idx_b, "codebook_bytes": cb_b,
                             "bits_per_weight": bits}
    print(f"B fcq_{variant}: acc={acc*100:.2f}% bits/weight={bits:.1f}", flush=True)

# ---------------- C. compression cost timing ----------------
r18b = resnet18_cifar()
r18b.load_state_dict(torch.load(os.path.join(OUT, "resnet18_best.pth"), map_location="cpu", weights_only=True))
t = {}
t0 = time.perf_counter(); f = fuse_bn(copy.deepcopy(r18b)); t["fuse_bn_s"] = time.perf_counter() - t0
for tau in [0.7, 0.8, 0.9]:
    tracemalloc.start()
    t0 = time.perf_counter(); d = decompose_convs(_fb(copy.deepcopy(r18b)), tau)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    t[f"decompose_tau{tau}_s"] = dt; t[f"decompose_tau{tau}_peak_mb"] = peak / 1e6
tracemalloc.start()
t0 = time.perf_counter(); mq, info = quantize_all_fc(copy.deepcopy(d))
t["fcq_all_linear_s"] = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
t["fcq_peak_mb"] = peak / 1e6
res["compression_cost"] = t
print("C cost:", json.dumps(t, default=float), flush=True)

# ---------------- D. actual rank ranges (seed 42, from recorded results.json) ----------------
rj = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
for model in ["resnet18"]:
    for tau, v in rj[model]["svd"].items():
        ks = v["k_per_layer"]
        res[f"krange_{model}_{tau}"] = {"min": min(ks), "max": max(ks),
                                        "mean": float(np.mean(ks)), "n_layers": len(ks)}
print("D krange:", json.dumps({k: v for k, v in res.items() if k.startswith("krange")}, default=float), flush=True)

# ---------------- E. footprint counterexample ----------------
def bytes_baseline(co, ci, hw): return co * ci * hw * 9 * 4
def bytes_decomposed(co, ci, hw, k): return (ci * hw * k * 9 + k * co) * 4
ce = {}
for (co, ci, k) in [(8, 64, 16), (16, 64, 32), (32, 32, 64), (8, 128, 16)]:
    b0, b1 = bytes_baseline(co, ci, 32), bytes_decomposed(co, ci, 32, k)
    ce[f"co{co}_ci{ci}_k{k}"] = {"baseline_bytes": b0, "decomposed_bytes": b1,
                                 "ratio": b0 / b1}
res["footprint_counterexample"] = ce
print("E counterexample:", json.dumps(ce, default=float), flush=True)

# ---------------- F. Table 3 exact bytes (KiB) ----------------
t3 = {}
for (co, ci) in [(256, 256), (512, 512)]:
    for k in [16, 32, 64, 96]:
        b0, b1 = bytes_baseline(co, ci, 32), bytes_decomposed(co, ci, 32, k)
        t3[f"{ci}x{co}_k{k}"] = {"baseline_bytes": b0, "decomposed_bytes": b1,
                                 "decomposed_KiB": b1 / 1024,
                                 "reduction": b0 / b1}
res["table3_exact"] = t3
print("F table3:", json.dumps(t3, default=float), flush=True)

with open(os.path.join(OUT, "round2_fixes.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, default=float)
print("ROUND2 FIXES DONE", flush=True)
