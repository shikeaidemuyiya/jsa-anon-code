# -*- coding: utf-8 -*-
"""Minimal fair comparison on CIFAR-10 at the SAME 3-epoch fine-tuning budget as the
existing decomposition results, so all methods are directly comparable:

  baseline (fused) | DepGraph @ratio0.3 (~2x FLOPs, matched to tau=0.7) | INT8 PTQ

Our tau=0.7/0.8/0.9 numbers come from results.json (same 3-epoch protocol).
"""
import copy, json, os
import torch

from baselines import get_loaders, measure, OUT, depgraph_prune, int8_static, make_model
from compress import fuse_bn

if __name__ == "__main__":
    tl, vl, in_size, nc = get_loaders("cifar10", workers=0)
    base_model = make_model("cifar10")
    res = {}

    b = base = measure(fuse_bn(copy.deepcopy(base_model)), vl, in_size)
    res["baseline"] = base
    print(f"baseline acc={base['acc']*100:.2f}% flops={base['flops']/1e6:.0f}M "
          f"lat_b8={base['lat_b8']:.2f}ms", flush=True)

    print("--- DepGraph r=0.3, 3 epochs (matched protocol) ---", flush=True)
    d = depgraph_prune(base_model, tl, vl, in_size, 0.30, 3, 0.02)
    res["depgraph_r03_3ep"] = d
    print(f"depgraph r0.3 acc={d['acc']*100:.2f}% flops_down={d['flops_reduction']:.2f}x "
          f"params={d['params']/1e6:.2f}M lat_b8={d['lat_b8']:.2f}ms", flush=True)
    with open(os.path.join(OUT, "baselines_cifar.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)

    print("--- DepGraph r=0.5, 3 epochs ---", flush=True)
    d5 = depgraph_prune(base_model, tl, vl, in_size, 0.50, 3, 0.02)
    res["depgraph_r05_3ep"] = d5
    print(f"depgraph r0.5 acc={d5['acc']*100:.2f}% flops_down={d5['flops_reduction']:.2f}x "
          f"lat_b8={d5['lat_b8']:.2f}ms", flush=True)

    print("--- INT8 PTQ ---", flush=True)
    q = int8_static(fuse_bn(copy.deepcopy(base_model)), tl, vl, in_size)
    res["int8"] = q
    print(f"int8 acc={q['acc']*100:.2f}% lat_b8={q['lat_b8']:.2f}ms "
          f"size={q.get('size_bytes_q', q['size_bytes'])/1e6:.2f}MB", flush=True)

    with open(os.path.join(OUT, "baselines_cifar.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print("MIN COMPARE DONE", flush=True)
