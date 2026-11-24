# -*- coding: utf-8 -*-
"""ImageNet-100 cross-dataset experiment (final protocol).

Protocol (identical recovery budget for all methods):
  1. BN folding
  2. hierarchical SVD on all 3x3 convs, 7x7 stem kept intact (standard practice:
     first-layer kernels are high-frequency and rank-truncation is destructive)
  3. CE recovery: AdamW lr=3e-4, cosine, N epochs
  4. KD polish:    AdamW lr=1e-4, T=2, alpha=0.5, 3 epochs (teacher = full model)
  5. stage-3 density-peak FC quantization (orthogonal add-on)

Baseline: DepGraph structured pruning (torch-pruning) with the same
          N CE epochs + 3 KD epochs.
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from baselines import get_loaders, measure, evaluate, DEVICE, OUT
from compress import fuse_bn, decompose_convs, collect_ks, LowRankConv
from run_experiments import quantize_all_fc, compressed_size

TEACHER_CKPT = os.path.join(OUT, "resnet18_imagenet100_best.pth")


def build_teacher():
    m = torchvision.models.resnet18(num_classes=100)
    m.load_state_dict(torch.load(TEACHER_CKPT, map_location="cpu", weights_only=True))
    return m


def decompose_skip_stem(model, tau):
    """Decompose every spatial conv EXCEPT the 7x7 stem (and depthwise convs)."""
    def rec(container):
        for name, m in list(container.named_children()):
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] > 1 and m.groups != m.in_channels:
                setattr(container, name, LowRankConv(m, tau))
            else:
                rec(m)
    for name, m in list(model.named_children()):
        if name == "conv1":                      # keep stem
            continue
        if isinstance(m, nn.Conv2d) and m.groups != m.in_channels:
            setattr(model, name, LowRankConv(m, tau))
        else:
            rec(m)
    return model


def train_ce(model, tl, vl, epochs, lr=3e-4, tag="", ckpt=None):
    model = model.to(DEVICE)
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        acc = evaluate(model, vl)
        print(f"  {tag} ce RESUME from {os.path.basename(ckpt)} acc={acc*100:.2f}%", flush=True)
        return model, acc
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))
    crit = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    for ep in range(epochs):
        t0 = time.time(); model.train()
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        acc = evaluate(model, vl)
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ckpt:
                torch.save({k: v.float().cpu() for k, v in best_state.items()}, ckpt)
        print(f"  {tag} ce ep{ep+1}/{epochs} acc={acc*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def train_kd(model, teacher, tl, vl, epochs=3, lr=1e-4, T=2.0, alpha=0.5, tag="", ckpt=None):
    model = model.to(DEVICE).train()
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        acc = evaluate(model, vl)
        print(f"  {tag} kd RESUME from {os.path.basename(ckpt)} acc={acc*100:.2f}%", flush=True)
        return model.cpu(), acc
    teacher = teacher.to(DEVICE).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))
    ce = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = model(x)
            loss = alpha * F.kl_div(F.log_softmax(s_logits / T, dim=1),
                                    F.softmax(t_logits / T, dim=1),
                                    reduction="batchmean") * T * T \
                   + (1 - alpha) * ce(s_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        acc = evaluate(model, vl)
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ckpt:
                torch.save({k: v.float().cpu() for k, v in best_state.items()}, ckpt)
        print(f"  {tag} kd ep{ep+1}/{epochs} acc={acc*100:.2f}%", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), best


def depgraph(model, ratio):
    """DepGraph-style structured pruning via torch-pruning (same as baselines)."""
    import torch_pruning as tp
    m = copy.deepcopy(model).to(DEVICE).eval()
    ex = torch.randn(1, 3, 224, 224).to(DEVICE)
    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(model=m, example_inputs=ex, importance=imp,
                                       iterative_steps=1, pruning_ratio=ratio,
                                       ignored_layers=[m.fc], global_pruning=True)
    pruner.step()
    return m.cpu()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="0.90")
    ap.add_argument("--ratios", default="0.30")
    ap.add_argument("--ce_epochs", type=int, default=12)
    ap.add_argument("--kd_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="results_imagenet100.json")
    args = ap.parse_args()

    tl, vl, in_size, nc = get_loaders("imagenet100")
    base = build_teacher()
    teacher = fuse_bn(copy.deepcopy(base)).eval()
    base_m = measure(copy.deepcopy(teacher), vl, in_size)
    print(f"teacher acc={base_m['acc']*100:.2f}% flops={base_m['flops']/1e9:.3f}G "
          f"params={base_m['params']/1e6:.2f}M", flush=True)

    res = {"teacher": base_m,
           "protocol": {"svd": f"skip-stem SVD -> AdamW {args.lr} x{args.ce_epochs} CE "
                               f"-> KD(AdamW 1e-4,T=2,a=0.5) x{args.kd_epochs} -> FCQ",
                        "depgraph": f"r -> AdamW {args.lr} x{args.ce_epochs} CE "
                                    f"-> KD(AdamW 1e-4,T=2,a=0.5) x{args.kd_epochs}"},
           "ours": {}, "depgraph": {}}

    def dump():
        with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=float)

    for tau in [float(t) for t in args.taus.split(",")]:
        tag = f"tau{tau}"
        print(f"=== {tag}: skip-stem SVD ===", flush=True)
        m = decompose_skip_stem(fuse_bn(copy.deepcopy(base)), tau)
        ks = collect_ks(m)
        acc0 = evaluate(m, vl)
        print(f"  acc_before_ft={acc0*100:.2f}% k_mean={np.mean(ks):.1f}", flush=True)
        m, acc_ce = train_ce(m, tl, vl, args.ce_epochs, args.lr, tag,
                            ckpt=os.path.join(OUT, f"in100_{tag}_ce.pth"))
        m, acc_kd = train_kd(m, teacher, tl, vl, args.kd_epochs, tag=tag,
                             ckpt=os.path.join(OUT, f"in100_{tag}_kd.pth"))
        m = m.cpu()
        st = measure(copy.deepcopy(m), vl, in_size)
        mq, fc_info = quantize_all_fc(copy.deepcopy(m))
        acc_q = evaluate(mq, vl)
        res["ours"][tag] = {"tau": tau, "k_mean": float(np.mean(ks)), "acc_before_ft": acc0,
                            "acc_ce": acc_ce, "acc_kd": acc_kd, **st,
                            "fcq": {"acc": acc_q, "size_bytes": compressed_size(mq, fc_info)}}
        dump()
        print(f"  {tag} RESULT acc={st['acc']*100:.2f}% (+FCQ {acc_q*100:.2f}%) "
              f"flops_down={base_m['flops']/st['flops']:.2f}x params={st['params']/1e6:.2f}M "
              f"lat_b1={st['lat_b1']:.3f}ms lat_b8={st['lat_b8']:.3f}ms", flush=True)

    for r in [float(x) for x in args.ratios.split(",")]:
        tag = f"r{r}"
        print(f"=== {tag}: DepGraph pruning ===", flush=True)
        m = depgraph(base, r)
        st0 = measure(copy.deepcopy(m), vl, in_size)
        print(f"  acc_before_ft={st0['acc']*100:.2f}% params={st0['params']/1e6:.2f}M", flush=True)
        m, acc_ce = train_ce(m, tl, vl, args.ce_epochs, args.lr, tag,
                            ckpt=os.path.join(OUT, f"in100_{tag}_ce.pth"))
        m, acc_kd = train_kd(m, teacher, tl, vl, args.kd_epochs, tag=tag,
                             ckpt=os.path.join(OUT, f"in100_{tag}_kd.pth"))
        m = m.cpu()
        st = measure(copy.deepcopy(m), vl, in_size)
        res["depgraph"][tag] = {"ratio": r, "acc_before_ft": st0["acc"],
                                "acc_ce": acc_ce, "acc_kd": acc_kd, **st}
        dump()
        print(f"  {tag} RESULT acc={st['acc']*100:.2f}% flops_down={base_m['flops']/st['flops']:.2f}x "
              f"params={st['params']/1e6:.2f}M lat_b8={st['lat_b8']:.3f}ms", flush=True)

    print("IMAGENET100 DONE", flush=True)
