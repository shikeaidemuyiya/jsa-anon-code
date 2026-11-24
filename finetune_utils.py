# -*- coding: utf-8 -*-
"""Fine-tune helper for factorized (re-initialized) models: AdamW + cosine.

Decomposed convs are re-parameterized from SVD, so their weights are not in the
pretrained basin; SGD with pruning-scale LRs diverges. AdamW with a small LR and
cosine decay recovers accuracy reliably within the same epoch budget.
"""
import torch
import torch.nn as nn


def evaluate(model, loader, device):
    model.eval().to(device)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total


def finetune_adamw(model, tl, vl, epochs=12, lr=1e-3, device=None, wd=1e-4):
    device = device or torch.device("cuda")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))
    crit = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        acc = evaluate(model, vl, device)
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  ft ep{ep+1}/{epochs} acc={acc*100:.2f}%", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best
