# -*- coding: utf-8 -*-
"""KD comparison: every compressed student gets its OWN best recovery recipe,
then the SAME KD polish on top (teacher = full baseline). Honest protocol:
per-method recommended fine-tuning is disclosed in the paper.

  SVD students:  decompose -> AdamW lr=1e-4, 3 epochs CE (reproduces 87-90%)
                 -> KD polish AdamW lr=5e-5, T=2, alpha=0.5, 3 epochs -> FCQ
  DepGraph:      prune -> SGD lr=0.02 cosine, 12 epochs (its best known recipe)
                 -> KD polish SGD lr=2e-3, T=2, alpha=0.5, 3 epochs
"""
import argparse, copy, json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

from baselines import get_loaders, make_model, DEVICE, OUT
from finetune_utils import evaluate
from compress import fuse_bn, decompose_convs
from run_experiments import quantize_all_fc
from supplement import count_flops

def build_pruned(base_model, in_size, ratio):
    m = copy.deepcopy(base_model).to(DEVICE)
    ex = torch.randn(1, 3, in_size, in_size).to(DEVICE)
    tp.pruner.MagnitudePruner(
        model=m, example_inputs=ex,
        importance=tp.importance.MagnitudeImportance(p=1),
        iterative_steps=1, pruning_ratio=ratio,
        ignored_layers=[m.fc], global_pruning=True).step()
    return m

def ce_adamw_lowlr(model, tl, vl, epochs=3, lr=1e-4, wd=5e-4):
    """Matches run_experiments.finetune recipe (the one that reached 87-90%)."""
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ce(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        acc = evaluate(model, vl, DEVICE)
        if acc > best:
            best = acc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    ce ep{ep+1}/{epochs} acc={acc*100:.2f}%", flush=True)
    model.load_state_dict(best_state)
    return model, best

def sgd_cosine(model, tl, vl, epochs=12, lr=0.02, wd=5e-4):
    model = model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))
    ce = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ce(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        acc = evaluate(model, vl, DEVICE)
        if acc > best:
            best = acc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    sgd ep{ep+1}/{epochs} acc={acc*100:.2f}%", flush=True)
    model.load_state_dict(best_state)
    return model, best

def kd_polish(model, teacher, tl, vl, epochs=3, lr=5e-5, T=2.0, alpha=0.5, wd=5e-4, opt_kind="adamw"):
    device = DEVICE
    model = model.to(device).train()
    teacher = teacher.to(device).eval()
    if opt_kind == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))
    ce = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = model(x)
            loss_kd = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                               F.softmax(t_logits / T, dim=1), reduction="batchmean") * T * T
            loss = alpha * loss_kd + (1 - alpha) * ce(s_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        acc = evaluate(model, vl, device)
        if acc > best:
            best = acc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    kd ep{ep+1}/{epochs} acc={acc*100:.2f}%", flush=True)
    model.load_state_dict(best_state)
    return model.cpu(), best

def stats(model, vl, in_size):
    acc = evaluate(model, vl, DEVICE)
    m_cpu = copy.deepcopy(model).cpu().eval()
    return {"acc": acc,
            "params": sum(p.numel() for p in model.parameters()),
            "flops": count_flops(m_cpu, in_shape=(1, 3, in_size, in_size))}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="0.70,0.80,0.90")
    ap.add_argument("--ratios", default="0.30,0.50")
    args = ap.parse_args()

    tl, vl, in_size, nc = get_loaders("cifar10", workers=0)
    base_model = make_model("cifar10")
    teacher = fuse_bn(copy.deepcopy(base_model)).eval()
    base_st = stats(teacher, vl, in_size)
    print(f"teacher acc={base_st['acc']*100:.2f}% flops={base_st['flops']/1e6:.0f}M", flush=True)

    res = {"teacher": base_st,
           "protocol": {"svd": "AdamW 1e-4 x3 CE -> KD(AdamW 5e-5, T=2, a=0.5) x3 -> FCQ",
                        "depgraph": "SGD 0.02 cosine x12 -> KD(SGD 2e-3, T=2, a=0.5) x3"},
           "students": {}}
    out_json = os.path.join(OUT, "kd_unified.json")

    for tau in [float(t) for t in args.taus.split(",")]:
        print(f"=== SVD tau={tau}: CE -> KD -> FCQ ===", flush=True)
        m = decompose_convs(fuse_bn(copy.deepcopy(base_model)), tau)
        st0 = stats(m, vl, in_size)
        m, acc_ce = ce_adamw_lowlr(m, tl, vl, epochs=3)
        m, acc_kd = kd_polish(m, teacher, tl, vl, epochs=3, lr=5e-5)
        mq, _ = quantize_all_fc(copy.deepcopy(m))
        st = stats(m, vl, in_size)
        stq = stats(mq, vl, in_size)
        st.update({"acc_before_ft": st0["acc"], "acc_ce": acc_ce, "acc_kd": acc_kd,
                   "acc_kd_fcq": stq["acc"]})
        res["students"][f"svd_tau{tau}"] = st
        torch.save(m.state_dict(), os.path.join(OUT, f"svd_tau{tau}_kd.pth"))
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=float)
        print(f"  tau={tau}: ce={acc_ce*100:.2f}% kd={acc_kd*100:.2f}% "
              f"kd+fcq={stq['acc']*100:.2f}% flops={st['flops']/1e6:.0f}M", flush=True)

    for ratio in [float(r) for r in args.ratios.split(",")]:
        print(f"=== DepGraph r={ratio}: SGD -> KD ===", flush=True)
        m = build_pruned(base_model, in_size, ratio)
        m, acc_ce = sgd_cosine(m, tl, vl, epochs=12, lr=0.02)
        m, acc_kd = kd_polish(m, teacher, tl, vl, epochs=3, lr=2e-3, opt_kind="sgd")
        st = stats(m, vl, in_size)
        st.update({"acc_sgd": acc_ce, "acc_kd": acc_kd})
        res["students"][f"depgraph_r{ratio}"] = st
        torch.save(m.state_dict(), os.path.join(OUT, f"depgraph_r{ratio}_kd.pth"))
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=float)
        print(f"  r={ratio}: sgd={acc_ce*100:.2f}% kd={acc_kd*100:.2f}% "
              f"flops={st['flops']/1e6:.0f}M params={st['params']/1e6:.2f}M", flush=True)

    print("KD DONE", flush=True)
