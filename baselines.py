# -*- coding: utf-8 -*-
"""SOTA baselines for the paper's comparison table, unified measurement protocol.

Methods:
  - DepGraph structured pruning (torch-pruning, magnitude importance), target FLOPs
    reduction matched to our tau=0.7 operating point (~52%), followed by fine-tuning.
  - INT8 post-training static quantization (torch.ao.quantization FX API).

Metrics for every method: top-1 acc, params, FLOPs, batch=1 & batch=8 single-thread
CPU latency, FP32-equivalent size (INT8 stored as uint8).
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch_pruning as tp

from compress import fuse_bn
from supplement import count_flops, batched_latency

OUT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda")

# ---------------------------------------------------------------- data
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class HFWrap(torch.utils.data.Dataset):
    """Wrap an HF Dataset with a transform; defined at module level for pickling on Windows."""
    def __init__(self, ds, tf): self.ds = ds; self.tf = tf
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[int(i)]
        return self.tf(r["image"]), r["label"]

def get_loaders(dataset, bs=256, workers=3):
    if dataset == "cifar10":
        tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
        te = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
        train = torchvision.datasets.CIFAR10(os.path.join(OUT, "data"), train=True, download=True, transform=tr)
        val = torchvision.datasets.CIFAR10(os.path.join(OUT, "data"), train=False, transform=te)
        in_size, num_classes = 32, 10
    elif dataset == "imagenet100":
        tr = T.Compose([T.Grayscale(num_output_channels=3), T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
        te = T.Compose([T.Grayscale(num_output_channels=3), T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
        from datasets import load_from_disk
        d = load_from_disk(os.path.join(OUT, "imagenet100"))
        train = HFWrap(d["train"], tr); val = HFWrap(d["validation"], te)
        in_size, num_classes = 224, 100
    else:
        raise ValueError(dataset)
    tl = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    vl = torch.utils.data.DataLoader(val, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return tl, vl, in_size, num_classes

# ---------------------------------------------------------------- models
def make_model(dataset, pretrained=False):
    if dataset == "cifar10":
        m = torchvision.models.resnet18(num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
        sd = torch.load(os.path.join(OUT, "resnet18_best.pth"), map_location="cpu", weights_only=True)
        m.load_state_dict(sd)
    else:
        m = torchvision.models.resnet18(num_classes=100)
        ckpt = os.path.join(OUT, "resnet18_imagenet100_best.pth")
        if os.path.exists(ckpt):
            m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        elif pretrained:
            sd = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
            sd["fc.weight"] = m.fc.weight.data; sd["fc.bias"] = m.fc.bias.data
            m.load_state_dict(sd)
    return m

# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, loader):
    model.eval().to(DEVICE)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total

def finetune(model, tl, epochs, lr, val_loader):
    model = model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step(); sched.step()
        acc = evaluate(model, val_loader)
        best = max(best, acc)
        print(f"  ft ep{ep+1}/{epochs} acc={acc*100:.2f}%", flush=True)
    return model, best

def measure(model, vl, in_size):
    acc = evaluate(model, vl)
    m_cpu = copy.deepcopy(model).cpu().eval()
    st_params = sum(p.numel() for p in model.parameters())
    flops = count_flops(m_cpu, in_shape=(1, 3, in_size, in_size))
    lat1 = batched_latency(model, 1, n_run=100)
    lat8 = batched_latency(model, 8, n_run=50)
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"acc": acc, "params": st_params, "flops": flops,
            "lat_b1": lat1, "lat_b8": lat8, "size_bytes": size}

# ---------------------------------------------------------------- baselines
def depgraph_prune(model, tl, vl, in_size, target_ratio, ft_epochs, ft_lr):
    """Structured pruning via torch-pruning magnitude importance (DepGraph-style)."""
    m = copy.deepcopy(model).to(DEVICE)          # keep BN for fine-tuning
    ex = torch.randn(1, 3, in_size, in_size).to(DEVICE)
    imp = tp.importance.MagnitudeImportance(p=1)
    ignored = [m.fc]
    pruner = tp.pruner.MagnitudePruner(model=m, example_inputs=ex, importance=imp,
                                       iterative_steps=1, pruning_ratio=target_ratio,
                                       ignored_layers=ignored, global_pruning=True)
    pruner.step()
    pruned_flops = count_flops(copy.deepcopy(m).cpu().eval(), (1, 3, in_size, in_size))
    base_flops = count_flops(copy.deepcopy(model).cpu().eval(), (1, 3, in_size, in_size))
    print(f"  pruned FLOPs: {pruned_flops/1e6:.0f}M ({base_flops/pruned_flops:.2f}x)", flush=True)
    m, acc = finetune(m, tl, ft_epochs, ft_lr, vl)
    m = fuse_bn(m.cpu())
    r = measure(m, vl, in_size)
    r["acc"] = acc
    r["flops_reduction"] = base_flops / pruned_flops
    return r

def int8_static(model, tl, vl, in_size):
    """FX post-training static quantization (calibrated on training data)."""
    m = copy.deepcopy(model).cpu().eval()        # BN-fused model, calibrate-free of BN stats
    m.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    from torch.ao.quantization import quantize_fx
    ex = (torch.randn(1, 3, in_size, in_size),)
    prepared = quantize_fx.prepare_fx(m, {"": torch.ao.quantization.default_qconfig}, example_inputs=ex)
    n_cal = 0
    with torch.no_grad():
        for x, _ in tl:
            prepared(x)
            n_cal += 1
            if n_cal >= 200: break
    q = quantize_fx.convert_fx(prepared)
    r = measure(q, vl, in_size)
    # size with uint8 weights
    total = 0
    for _, p in q.state_dict().items():
        total += p.numel() * p.element_size()
    r["size_bytes_q"] = total
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "imagenet100"])
    ap.add_argument("--prune_ratio", type=float, default=0.5)
    ap.add_argument("--ft_epochs", type=int, default=12)
    ap.add_argument("--ft_lr", type=float, default=0.02)
    ap.add_argument("--skip_int8", action="store_true")
    args = ap.parse_args()

    tl, vl, in_size, nc = get_loaders(args.dataset)
    base_model = make_model(args.dataset)
    print(f"=== dataset={args.dataset} ===", flush=True)
    res = {"dataset": args.dataset}

    base = measure(fuse_bn(copy.deepcopy(base_model)), vl, in_size)
    res["baseline"] = base
    print(f"baseline acc={base['acc']*100:.2f}% params={base['params']} "
          f"flops={base['flops']/1e6:.0f}M lat_b8={base['lat_b8']:.3f}ms", flush=True)

    print("--- DepGraph pruning ---", flush=True)
    res["depgraph"] = depgraph_prune(base_model, tl, vl, in_size, args.prune_ratio,
                                     args.ft_epochs, args.ft_lr)
    d = res["depgraph"]
    print(f"depgraph acc={d['acc']*100:.2f}% flops_down={d['flops_reduction']:.2f}x "
          f"lat_b8={d['lat_b8']:.3f}ms", flush=True)

    if not args.skip_int8:
        print("--- INT8 static quantization ---", flush=True)
        res["int8"] = int8_static(fuse_bn(copy.deepcopy(base_model)), tl, vl, in_size)
        print(f"int8 acc={res['int8']['acc']*100:.2f}% lat_b8={res['int8']['lat_b8']:.3f}ms", flush=True)

    with open(os.path.join(OUT, f"baselines_{args.dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print("BASELINES DONE", flush=True)
