# -*- coding: utf-8 -*-
"""Run Patent-1 experiments: accuracy / size / CPU latency / cache simulation / ablations."""
import json, os, time, copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from compress import (fuse_bn, decompose_convs, choose_K_adaptive, cluster_quantize,
                      QuantizedFC, diff_encode_bias, simulate_cache_weight_access,
                      model_stats, layout_depthwise, layout_pointwise, CACHE_LINE,
                      collect_ks)

OUT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda")

def resnet18_cifar(num_classes=10):
    import torch.nn as nn_
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn_.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn_.Identity()
    return m

def mobilenetv2_cifar(num_classes=10):
    m = torchvision.models.mobilenet_v2(num_classes=num_classes)
    m.features[0][0].stride = (1, 1)
    return m

MODELS = {"resnet18": resnet18_cifar, "mobilenetv2": mobilenetv2_cifar}

def get_test_loader(bs=512):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    te = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test = torchvision.datasets.CIFAR10(os.path.join(OUT, "data"), train=False, download=True, transform=te)
    return torch.utils.data.DataLoader(test, batch_size=bs, shuffle=False, num_workers=4)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval().to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total

@torch.no_grad()
def cpu_latency(model, n_warm=30, n_run=200):
    """Single-thread CPU latency (edge-like), batch=1, 3x32x32 input."""
    model = copy.deepcopy(model).cpu().eval()
    torch.set_num_threads(1)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for _ in range(n_warm): model(x)
        ts = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.median(ts)), float(np.std(ts))

def quantize_all_fc(model, cache_bytes=32*1024):
    """Adaptive density-peak quantization of every Linear layer (patent step 3)."""
    info = []
    def _recurse(container):
        for name, m in list(container.named_children()):
            if isinstance(m, nn.Linear):
                sel = choose_K_adaptive(m.weight.data.numpy(), cache_bytes=cache_bytes)
                if sel is None:
                    centers, a, _ = cluster_quantize(m.weight.data.numpy(), 64)
                    sel = {"K": 64, "used_clusters": 64, "coverage": 1.0, "mse": 0.0, "dc_mult": 1.0}
                centers, idx, counts = cluster_quantize(m.weight.data.numpy(), sel["K"])
                qfc = QuantizedFC(m, centers, idx)
                setattr(container, name, qfc)
                info.append({**sel, "in_f": m.in_features, "out_f": m.out_features,
                             "quant_bytes": idx.size * 2 + len(centers) * 4,
                             "orig_bytes": m.weight.numel() * 4,
                             "bias_coding": diff_encode_bias(m.bias.data.numpy()) if m.bias is not None else None})
            else:
                _recurse(m)
    _recurse(model)
    return model, info

def compressed_size(model, fc_info):
    """f32 conv/decomposed weights + quantized FC (uint16 idx + codebook) + coded biases."""
    total = 0
    for n, p in model.named_parameters():
        if "codebook" in n or "idx" in n: continue
        total += p.numel() * 4
    for r in fc_info:
        total += r["quant_bytes"]
        if r["bias_coding"]:
            total += r["bias_coding"]["bias_bytes_coded"]
        else:
            total += r["out_f"] * 4
    return total

def finetune(model, epochs=3, lr=1e-4):
    """Brief accuracy-recovery fine-tune on CIFAR-10 train set (GPU)."""
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    train = torchvision.datasets.CIFAR10(os.path.join(OUT, "data"), train=True, transform=tr)
    tl = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
    return model

def cache_experiment():
    """Weight DRAM traffic per layer invocation: baseline conv vs decomposed (pw1 1x1 + pw2 spatial),
    under cache-line aligned layout (patent step 2b)."""
    from compress import weight_traffic_baseline, weight_traffic_decomposed
    res = {}
    # representative layers of a CIFAR CNN (out_ch, in_ch, spatial hw)
    layers = {"conv_64x64x9": (64, 64, 9), "conv_128x128x9": (128, 128, 9),
              "conv_256x256x9": (256, 256, 9), "conv_512x512x9": (512, 512, 9)}
    for lname, (cout, cin, hw) in layers.items():
        row = {}
        for k in [16, 32, 64, 96]:
            base = weight_traffic_baseline(cout, cin, hw)
            dec = weight_traffic_decomposed(cout, cin, hw, k)
            row[f"k={k}"] = {"baseline_bytes": base, "decomposed_bytes": dec,
                             "traffic_reduction": base / dec}
        res[lname] = row
    # LRU cache miss simulation under oc-major loop (weights resident per oc)
    for cache_kb in [16, 32]:
        r_base = simulate_cache_weight_access(128 * 9 * 4, 256, 100, cache_size=cache_kb * 1024)
        r_pw = simulate_cache_weight_access(4 * 9 * 4, 256, 100, cache_size=cache_kb * 1024)
        res[f"lru_{cache_kb}KB"] = {
            "baseline_miss_rate": r_base["miss_rate"],
            "decomposed_pw_miss_rate": r_pw["miss_rate"],
            "baseline_traffic": r_base["dram_weight_traffic_bytes"],
            "decomposed_traffic": r_pw["dram_weight_traffic_bytes"]}
    return res

def run_model(name, loader, results):
    model = MODELS[name]()
    model.load_state_dict(torch.load(os.path.join(OUT, f"{name}_best.pth"), map_location="cpu", weights_only=True))
    base = {}
    base["acc"] = evaluate(model, loader, DEVICE)
    fused = fuse_bn(model)
    base["acc_fused"] = evaluate(fused, loader, DEVICE)
    base.update(model_stats(fused))
    lat, med, sd = cpu_latency(fused)
    base["latency_ms"] = lat; base["latency_med_ms"] = med; base["latency_std_ms"] = sd
    print(f"[{name}] base acc={base['acc']*100:.2f}% fused={base['acc_fused']*100:.2f}% "
          f"params={base['params']} latency={lat:.2f}ms", flush=True)
    results[name] = {"baseline": base, "svd": {}}

    for tau in [0.70, 0.80, 0.90]:
        m = MODELS[name]()
        m.load_state_dict(torch.load(os.path.join(OUT, f"{name}_best.pth"), map_location="cpu", weights_only=True))
        m = fuse_bn(m)
        m = decompose_convs(m, tau)
        ks = collect_ks(m)
        acc0 = evaluate(m, loader, DEVICE)
        m = finetune(m, epochs=3)
        acc = evaluate(m, loader, DEVICE)
        st = model_stats(m)
        lat, med, sd = cpu_latency(m)
        r = {"tau": tau, "k_per_layer": ks, "k_mean": float(np.mean(ks)),
             "acc_before_ft": acc0, "acc_after_ft": acc, **st,
             "latency_ms": lat, "latency_med_ms": med, "latency_std_ms": sd}
        # then quantize FC on top
        mq, fc_info = quantize_all_fc(copy.deepcopy(m).cpu())
        accq = evaluate(mq, loader, DEVICE)
        latq, medq, sdq = cpu_latency(mq)
        size_q = compressed_size(mq, fc_info)
        r["fc_quant"] = {"acc": accq, "latency_ms": latq, "size_bytes": size_q, "layers": fc_info}
        results[name]["svd"][str(tau)] = r
        with open(os.path.join(OUT, "results_partial.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=float)
        print(f"[{name}] tau={tau} acc0={acc0*100:.2f}% acc_ft={acc*100:.2f}% "
              f"params={st['params']} lat={lat:.2f}ms | +FCQ acc={accq*100:.2f}% lat={latq:.2f}ms "
              f"size={size_q/1e6:.2f}MB", flush=True)

if __name__ == "__main__":
    loader = get_test_loader()
    results = {}
    for name in MODELS:
        run_model(name, loader, results)
    results["cache_simulation"] = cache_experiment()
    # FC bias diff-coding standalone stats
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print("EXPERIMENTS DONE", flush=True)
