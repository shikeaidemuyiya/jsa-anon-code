# -*- coding: utf-8 -*-
"""Round-3 fixes:
A. Event-level per-channel DMA staging simulation (aligned vs packed layout)
   on the 256x256 3x3 layer at k=4/7/13/16, both stages.
B. Relative rank k/min(Co, Ci*9) distribution over the 17 decomposed layers.
C. Footprint-aware rank cap: how many selected ranks exceed the
   no-footprint-growth cap, and what capping would change (statistics only).
Outputs: round3_fixes.json
"""
import os, sys, json
import numpy as np
import torch

HERE = r"."
sys.path.insert(0, HERE)
os.chdir(HERE)

from compress import fuse_bn, decompose_convs, collect_ks  # noqa: E402
from train_baseline import resnet18_cifar  # noqa: E402

LINE = 64
res = {}

# ---------------- A. DMA staging event-level simulation ----------------
def stage_sim(block_bytes, n_channels):
    """Per-channel DMA staging of one stage's weights.

    aligned-per-channel: channel i block starts at i*ceil(B/64 rounded to
    line), i.e. one padded slot per channel, fixed stride.
    packed: blocks tightly concatenated, start_i = i*B.
    DMA fetches whole lines; a request covers ceil/span lines of the block.
    Returns dict with lines fetched, overfetch bytes, buffer bytes.
    """
    slot = ((block_bytes + LINE - 1) // LINE) * LINE
    # aligned
    a_lines_per_ch = slot // LINE
    a_lines = a_lines_per_ch * n_channels
    a_overfetch = a_lines * LINE - block_bytes * n_channels
    a_buffer = slot * n_channels
    # packed
    p_lines = 0
    for i in range(n_channels):
        start = i * block_bytes
        end = start + block_bytes - 1
        span = end // LINE - start // LINE + 1
        p_lines += span
    p_overfetch = p_lines * LINE - block_bytes * n_channels
    p_buffer = block_bytes * n_channels
    return {
        "block_bytes": block_bytes, "channels": n_channels,
        "aligned": {"lines_per_channel": a_lines_per_ch, "total_lines": a_lines,
                    "overfetch_bytes": a_overfetch, "staging_buffer_bytes": a_buffer,
                    "fixed_stride": True, "block_straddles_line": False},
        "packed": {"total_lines": p_lines,
                   "straddling_channels": sum(
                       1 for i in range(n_channels)
                       if ((i * block_bytes + block_bytes - 1) // LINE
                           != (i * block_bytes) // LINE)),
                   "overfetch_bytes": p_overfetch,
                   "staging_buffer_bytes": p_buffer,
                   "fixed_stride": False},
    }

sim = {}
# pointwise stage k->Co: block = k*4 bytes per output channel, Co=256
for k in [4, 7, 13, 16]:
    sim[f"pointwise_k{k}"] = stage_sim(k * 4, 256)
# spatial stage Ci->k: block = Ci*9*4 bytes, Ci=256
for k in [4, 16]:
    sim[f"spatial_Ci256_k{k}"] = stage_sim(256 * 9 * 4, 256)
# spatial stage at Ci=64 (small layer)
sim["spatial_Ci64_k16"] = stage_sim(64 * 9 * 4, 256)
res["dma_staging"] = sim
print("A dma:", json.dumps(sim, indent=1), flush=True)

# ---------------- B. relative ranks ----------------
sd = torch.load(os.path.join(HERE, "resnet18_best.pth"), map_location="cpu",
                weights_only=True)
rel = {}
from compress import LowRankConv
for tau in [0.7, 0.8, 0.9]:
    m = resnet18_cifar()
    m.load_state_dict(sd)
    d = decompose_convs(fuse_bn(m), tau)
    rows = []
    for n, mod in d.named_modules():
        if isinstance(mod, LowRankConv):
            ci = mod.s3.in_channels
            co = mod.p1.out_channels
            upper = min(co, ci * 9)
            rows.append({"k": mod.k, "co": co, "ci": ci,
                         "upper": upper, "rel": mod.k / upper})
    ks = [r["k"] for r in rows]
    rels = [r["rel"] for r in rows]
    rel[f"tau{tau}"] = {
        "n_layers": len(rows), "k_min": min(ks), "k_max": max(ks),
        "rel_min": round(min(rels), 3), "rel_max": round(max(rels), 3),
        "rel_mean": round(float(np.mean(rels)), 3),
        "rel_median": round(float(np.median(rels)), 3),
        "detail": rows,
    }
    print(f"B tau={tau}: rel range {min(rels):.2f}-{max(rels):.2f} "
          f"mean {np.mean(rels):.2f}", flush=True)
res["relative_rank"] = rel

# ---------------- C. footprint-aware rank cap ----------------
# cap: k <= 9*Co*Ci / (9*Ci + Co)  keeps two-stage footprint <= baseline
cap_stats = {}
for tau in [0.7, 0.8, 0.9]:
    rows = rel[f"tau{tau}"]["detail"]
    n_over, capped_ks = 0, []
    for r in rows:
        co, ci, k = r["co"], r["ci"], r["k"]
        kcap = 9 * co * ci / (9 * ci + co)
        if k > kcap:
            n_over += 1
            capped_ks.append({"k": k, "cap": round(kcap, 1),
                              "co": co, "ci": ci})
    cap_stats[f"tau{tau}"] = {
        "n_layers": len(rows), "n_layers_over_cap": n_over,
        "over_cap_layers": capped_ks,
        "baseline_floors_rel": round(
            float(np.mean([r["k"] / r["upper"] for r in rows])), 3),
    }
    print(f"C tau={tau}: {n_over}/{len(rows)} layers over cap", flush=True)
res["rank_cap"] = cap_stats

with open(os.path.join(HERE, "round3_fixes.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, default=float)
print("ROUND3 DONE", flush=True)
