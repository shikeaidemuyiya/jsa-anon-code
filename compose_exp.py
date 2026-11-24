# -*- coding: utf-8 -*-
"""Composition experiment: DepGraph pruning -> our low-rank decomposition -> FC clustering
quantization, with cache-line-aligned layout. Tests whether the patent pipeline is
orthogonal to structured pruning (and therefore still worth publishing)."""
import copy, json, os
import torch, torch.nn as nn
import torch_pruning as tp

from baselines import get_loaders, measure, OUT, make_model
from finetune_utils import finetune_adamw
from compress import fuse_bn, decompose_convs, collect_ks
from run_experiments import quantize_all_fc, compressed_size
from supplement import count_flops

DEVICE = torch.device("cuda")

def prune(model, in_size, ratio=0.30):
    m = copy.deepcopy(model).to(DEVICE)
    ex = torch.randn(1, 3, in_size, in_size).to(DEVICE)
    pruner = tp.pruner.MagnitudePruner(model=m, example_inputs=ex,
                                       importance=tp.importance.MagnitudeImportance(p=1),
                                       iterative_steps=1, pruning_ratio=ratio,
                                       ignored_layers=[m.fc], global_pruning=True)
    pruner.step()
    return m

def sgd_ft(m, tl, vl, epochs, lr=0.02):
    m = m.to(DEVICE)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train()
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(m(x), y).backward(); opt.step(); sched.step()
        print(f"  sgd ep{ep+1}/{epochs}", flush=True)
    return m

if __name__ == "__main__":
    tl, vl, in_size, nc = get_loaders("cifar10", workers=0)
    base = make_model("cifar10")
    res = {}

    # stage 1: reuse checkpoint if available (pruned r=0.3 + 3 epochs, 90.02%)
    ck = os.path.join(OUT, "depgraph_r03_ft3.pth")
    if os.path.exists(ck):
        # pruning is deterministic: rebuild the pruned architecture, then load weights
        m = prune(base, in_size, 0.30)
        m = fuse_bn(m.cpu())
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    else:
        m = prune(base, in_size, 0.30)
        m = sgd_ft(m, tl, vl, 3)
        m = fuse_bn(m.cpu())
        torch.save(m.state_dict(), ck)
    r1 = measure(m, vl, in_size)
    res["prune_only"] = r1
    print(f"prune r0.3 +3ep acc={r1['acc']*100:.2f}% flops={r1['flops']/1e6:.0f}M "
          f"params={r1['params']/1e6:.2f}M lat_b8={r1['lat_b8']:.2f}ms", flush=True)
    torch.save(m.state_dict(), os.path.join(OUT, "depgraph_r03_ft3.pth"))

    # stage 2: + our decomposition (higher tau to preserve accuracy) + FC quant
    for tau in [0.80, 0.90]:
        md = decompose_convs(copy.deepcopy(m).cpu(), tau)
        acc0 = measure(md, vl, in_size)["acc"]
        md, acc = finetune_adamw(md, tl, vl, epochs=3, lr=8e-4, device=DEVICE)
        md = md.cpu()
        r2 = {**measure(md, vl, in_size), "acc_before_ft": acc0, "acc_after_ft": acc,
              "k_mean": float(sum(collect_ks(md)) / max(1, len(collect_ks(md))))}
        mq, fc_info = quantize_all_fc(copy.deepcopy(md).cpu())
        r2["fc_quant"] = {"acc": measure(mq, vl, in_size)["acc"],
                          "size_bytes": compressed_size(mq, fc_info)}
        print(f"prune+decomp tau={tau}: acc0={acc0*100:.2f}% acc={r2['acc_after_ft']*100:.2f}% "
              f"flops={r2['flops']/1e6:.0f}M params={r2['params']/1e6:.2f}M "
              f"lat_b8={r2['lat_b8']:.2f}ms", flush=True)
        res[f"prune+decomp_{tau}"] = r2
        with open(os.path.join(OUT, "compose_cifar.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=float)
        print(f"prune+decomp tau={tau}: acc0={acc0*100:.2f}% acc={r2['acc_after_ft']*100:.2f}% "
              f"flops={r2['flops']/1e6:.0f}M params={r2['params']/1e6:.2f}M "
              f"lat_b8={r2['lat_b8']:.2f}ms | +FCQ acc={r2['fc_quant']['acc']*100:.2f}% "
              f"size={r2['fc_quant']['size_bytes']/1e6:.2f}MB", flush=True)
        with open(os.path.join(OUT, "compose_cifar.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=float)

    with open(os.path.join(OUT, "compose_cifar.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print("COMPOSE DONE", flush=True)
