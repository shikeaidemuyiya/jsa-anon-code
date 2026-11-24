# -*- coding: utf-8 -*-
"""Fair-protocol CIFAR-10 comparison: our pipeline re-fine-tuned with the SAME
protocol as the pruning baseline (SGD 0.02 cosine, 12 epochs), plus DepGraph at
FLOPs-matched ratio and INT8. Produces apples-to-apples numbers.
"""
import argparse, copy, json, os
import numpy as np
import torch

from baselines import get_loaders, finetune, measure, DEVICE, OUT, depgraph_prune, int8_static, make_model
from finetune_utils import finetune_adamw
from compress import fuse_bn, decompose_convs, collect_ks
from run_experiments import quantize_all_fc, compressed_size

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="train/eval on CPU (GPU busy with other jobs)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--taus", default="0.70,0.80,0.90")
    args = ap.parse_args()
    if args.cpu:
        import baselines
        baselines.DEVICE = torch.device("cpu")
        torch.set_num_threads(12)
    tl, vl, in_size, nc = get_loaders("cifar10", workers=0)
    base_model = make_model("cifar10")

    res = {}
    base = measure(fuse_bn(copy.deepcopy(base_model)), vl, in_size)
    res["baseline"] = base
    print(f"baseline acc={base['acc']*100:.2f}% flops={base['flops']/1e6:.0f}M "
          f"lat_b8={base['lat_b8']:.3f}ms", flush=True)

    # ---- our method, fair protocol
    res["ours"] = {}
    for tau in [float(t) for t in args.taus.split(",")]:
        print(f"decomposing tau={tau} ...", flush=True)
        m = fuse_bn(copy.deepcopy(base_model))
        m = decompose_convs(m, tau)
        acc0 = measure(m, vl, in_size)["acc"]
        m, acc = finetune_adamw(m, tl, vl, epochs=args.epochs, lr=1.5e-3,
                                device=torch.device("cpu") if args.cpu else None)
        m = m.cpu()
        r = {"tau": tau, "k_mean": float(np.mean(collect_ks(m))), "acc_before_ft": acc0,
             "acc_after_ft": acc, **measure(m, vl, in_size)}
        mq, fc_info = quantize_all_fc(copy.deepcopy(m))
        accq = measure(mq, vl, in_size)
        size_q = compressed_size(mq, fc_info)
        r["fc_quant"] = {"acc": accq["acc"], "lat_b1": accq["lat_b1"], "lat_b8": accq["lat_b8"],
                         "size_bytes": size_q}
        res["ours"][str(tau)] = r
        with open(os.path.join(OUT, "fair_cifar_partial.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=float)
        print(f"ours tau={tau} acc0={acc0*100:.2f}% acc_ft={acc*100:.2f}% "
              f"flops_down={base['flops']/r['flops']:.2f}x lat_b8={r['lat_b8']:.3f}ms "
              f"| +FCQ acc={accq['acc']*100:.2f}%", flush=True)

    # ---- DepGraph at FLOPs-matched ratio (target ~2x reduction, like tau=0.7)
    print("--- DepGraph ratio=0.30 (FLOPs-matched) ---", flush=True)
    res["depgraph_matched"] = depgraph_prune(base_model, tl, vl, in_size, 0.30, args.epochs, 0.02)
    d = res["depgraph_matched"]
    print(f"depgraph r=0.3 acc={d['acc']*100:.2f}% flops_down={d['flops_reduction']:.2f}x "
          f"lat_b8={d['lat_b8']:.3f}ms", flush=True)
    with open(os.path.join(OUT, "fair_cifar_partial.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)

    # ---- INT8
    print("--- INT8 ---", flush=True)
    res["int8"] = int8_static(fuse_bn(copy.deepcopy(base_model)), tl, vl, in_size)
    print(f"int8 acc={res['int8']['acc']*100:.2f}% lat_b8={res['int8']['lat_b8']:.3f}ms", flush=True)

    with open(os.path.join(OUT, "fair_results_cifar.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print("FAIR CIFAR DONE", flush=True)
