# -*- coding: utf-8 -*-
"""Run our compression pipeline (hierarchical SVD + FC quant) on ImageNet-100."""
import argparse, copy, json, os
import numpy as np
import torch

from baselines import get_loaders, measure, DEVICE, OUT
from finetune_utils import finetune_adamw
from compress import fuse_bn, decompose_convs, collect_ks
from run_experiments import quantize_all_fc, compressed_size

def build_model():
    import torchvision
    m = torchvision.models.resnet18(num_classes=100)
    m.load_state_dict(torch.load(os.path.join(OUT, "resnet18_imagenet100_best.pth"),
                                 map_location="cpu", weights_only=True))
    return m

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="0.70,0.80,0.90")
    ap.add_argument("--ft_epochs", type=int, default=6)
    ap.add_argument("--ft_lr", type=float, default=1e-3)
    args = ap.parse_args()

    tl, vl, in_size, nc = get_loaders("imagenet100")
    base = build_model()
    base_m = measure(fuse_bn(copy.deepcopy(base)), vl, in_size)
    print(f"baseline acc={base_m['acc']*100:.2f}% flops={base_m['flops']/1e9:.2f}G", flush=True)

    results = {"baseline": base_m, "svd": {}}
    for tau in [float(t) for t in args.taus.split(",")]:
        m = fuse_bn(copy.deepcopy(base))
        m = decompose_convs(m, tau)
        ks = collect_ks(m)
        acc0 = measure(m, vl, in_size)["acc"]
        m, acc = finetune_adamw(m, tl, vl, epochs=args.ft_epochs, lr=args.ft_lr)
        m = m.cpu()
        r = {"tau": tau, "k_mean": float(np.mean(ks)), "acc_before_ft": acc0, "acc_after_ft": acc,
             **measure(m, vl, in_size)}
        mq, fc_info = quantize_all_fc(copy.deepcopy(m))
        accq = measure(mq, vl, in_size)
        size_q = compressed_size(mq, fc_info)
        r["fc_quant"] = {"acc": accq["acc"], "lat_b1": accq["lat_b1"], "lat_b8": accq["lat_b8"],
                         "size_bytes": size_q}
        results["svd"][str(tau)] = r
        with open(os.path.join(OUT, "results_imagenet100_partial.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"tau={tau} acc0={acc0*100:.2f}% acc_ft={acc*100:.2f}% params={r['params']} "
              f"flops_down={base_m['flops']/r['flops']:.2f}x lat_b8={r['lat_b8']:.3f}ms "
              f"| +FCQ acc={accq['acc']*100:.2f}% size={size_q/1e6:.2f}MB", flush=True)

    with open(os.path.join(OUT, "results_imagenet100.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print("IMAGENET100 EXPERIMENTS DONE", flush=True)
