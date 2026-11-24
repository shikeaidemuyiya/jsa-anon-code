# -*- coding: utf-8 -*-
"""Multi-seed repetition of the full CIFAR-10 pipeline (review M4/m12).

For each seed in {0, 1, 2} (seed 42 exists in kd_unified.json):
  1. Train ResNet-18 (CIFAR variant) 20 ep AdamW lr=1e-3  -> teacher
  2. SVD decompose tau in {0.7, 0.8, 0.9}
       -> CE AdamW lr=1e-4 x3 -> KD polish (AdamW 5e-5, T=2, a=0.5) x3 -> FCQ
  3. DepGraph r in {0.3, 0.5}: SGD 0.02 cosine x12 -> KD (SGD 2e-3) x3
  4. INT8 static PTQ of the teacher
Results appended incrementally to multiseed_results.json.
"""
import copy, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torchvision

from baselines import get_loaders, DEVICE, OUT, int8_static
from finetune_utils import evaluate
from kd_exp import ce_adamw_lowlr, kd_polish, build_pruned, stats
from compress import fuse_bn, decompose_convs
from run_experiments import quantize_all_fc, compressed_size, cpu_latency
from train_baseline import resnet18_cifar

SEEDS = [0, 1, 2]
TAUS = [0.70, 0.80, 0.90]
RATIOS = [0.30, 0.50]
OUT_JSON = os.path.join(OUT, "multiseed_results.json")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_baseline(seed, tl, vl, epochs=20, lr=1e-3):
    """Same recipe as train_baseline.py (AdamW 1e-3, cosine, wd 5e-4)."""
    model = resnet18_cifar(num_classes=10).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
        sched.step()
        acc = evaluate(model, vl, DEVICE)
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [seed{seed}] baseline ep{ep}/{epochs} acc={acc*100:.2f}%", flush=True)
    model.load_state_dict(best_state)
    return model.cpu(), best


def run_seed(seed, results):
    seed_everything(seed)
    tl, vl, in_size, nc = get_loaders("cifar10", workers=0)

    entry = results.get(str(seed), {})
    entry["seed"] = seed

    # ---- 1. baseline / teacher
    if "baseline" not in entry:
        t0 = time.time()
        base_model, best = train_baseline(seed, tl, vl)
        entry["baseline"] = {"acc": best}
        fused = fuse_bn(copy.deepcopy(base_model))
        entry["baseline"]["acc_fused"] = evaluate(fused, vl, DEVICE)
        print(f"== seed {seed}: baseline best={best*100:.2f}% fused={entry['baseline']['acc_fused']*100:.2f}% "
              f"({time.time()-t0:.0f}s) ==", flush=True)
        entry["baseline"]["state_file"] = os.path.join(OUT, f"resnet18_seed{seed}_best.pth")
        torch.save(base_model.state_dict(), entry["baseline"]["state_file"])
        results[str(seed)] = entry
        dump(results)

    base_model = resnet18_cifar()
    base_model.load_state_dict(torch.load(entry["baseline"]["state_file"],
                                          map_location="cpu", weights_only=True))
    teacher = fuse_bn(copy.deepcopy(base_model)).eval()

    # ---- 2. our pipeline per tau
    entry.setdefault("ours", {})
    for tau in TAUS:
        key = f"tau{tau:.2f}"
        if key in entry["ours"]:
            continue
        print(f"== seed {seed}: SVD tau={tau} ==", flush=True)
        m = decompose_convs(fuse_bn(copy.deepcopy(base_model)), tau)
        st0 = stats(m, vl, in_size)
        m, acc_ce = ce_adamw_lowlr(m, tl, vl, epochs=3)
        m, acc_kd = kd_polish(m, teacher, tl, vl, epochs=3, lr=5e-5)
        st = stats(m, vl, in_size)
        mq, fc_info = quantize_all_fc(copy.deepcopy(m).cpu())
        acc_fcq = evaluate(mq, vl, DEVICE)
        size_q = compressed_size(mq, fc_info)
        r = {"acc_before_ft": st0["acc"], "acc_ce": acc_ce,
             "acc_kd": acc_kd, "acc_kd_fcq": acc_fcq,
             "k_mean": float(np.mean([q["K"] for q in fc_info])),
             "flops": st["flops"], "params": st["params"],
             "size_bytes": size_q}
        if abs(tau - 0.70) < 1e-6:
            lat, med, sd = cpu_latency(mq)
            r["lat_b1"] = {"lat_ms": lat, "med_ms": med, "std_ms": sd}
        entry["ours"][key] = r
        results[str(seed)] = entry
        dump(results)
        print(f"  tau={tau}: ce={acc_ce*100:.2f}% kd={acc_kd*100:.2f}% "
              f"fcq={acc_fcq*100:.2f}% size={size_q/1e6:.2f}MB", flush=True)

    # ---- 3. DepGraph
    entry.setdefault("depgraph", {})
    for ratio in RATIOS:
        key = f"r{ratio:.2f}"
        if key in entry["depgraph"]:
            continue
        print(f"== seed {seed}: DepGraph r={ratio} ==", flush=True)
        m = build_pruned(base_model, in_size, ratio)
        m, acc_sgd = sgd_cosine12(m, tl, vl)
        m, acc_kd = kd_polish(m, teacher, tl, vl, epochs=3, lr=2e-3, opt_kind="sgd")
        st = stats(m, vl, in_size)
        entry["depgraph"][key] = {"acc_sgd": acc_sgd, "acc_kd": acc_kd,
                                  "flops": st["flops"], "params": st["params"]}
        results[str(seed)] = entry
        dump(results)
        print(f"  r={ratio}: sgd={acc_sgd*100:.2f}% kd={acc_kd*100:.2f}%", flush=True)

    # ---- 4. INT8 (QConfigMapping API; legacy dict qconfig segfaults on torch 2.14)
    if "int8" not in entry:
        print(f"== seed {seed}: INT8 ==", flush=True)
        from torch.ao.quantization import quantize_fx, QConfigMapping, get_default_qconfig
        m8 = fuse_bn(copy.deepcopy(base_model)).cpu().eval()
        qm = QConfigMapping().set_global(get_default_qconfig("fbgemm"))
        prepared = quantize_fx.prepare_fx(m8, qm,
                                          example_inputs=(torch.randn(1, 3, in_size, in_size),))
        with torch.no_grad():
            for i, (x, _) in enumerate(tl):
                prepared(x)
                if i >= 50:
                    break
        q = quantize_fx.convert_fx(prepared).eval()
        acc8 = evaluate(q, vl, torch.device("cpu"))
        entry["int8"] = {"acc": acc8}
        results[str(seed)] = entry
        dump(results)

    print(f"SEED {seed} DONE", flush=True)


def sgd_cosine12(model, tl, vl):
    """SGD 0.02 cosine x12 — DepGraph recovery recipe from kd_exp.sgd_cosine."""
    from kd_exp import sgd_cosine
    return sgd_cosine(model, tl, vl, epochs=12, lr=0.02)


def dump(results):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            results = json.load(f)
    for seed in SEEDS:
        run_seed(seed, results)
    print("MULTISEED DONE", flush=True)
