# -*- coding: utf-8 -*-
"""Supplementary: FLOPs counting + batched CPU throughput latency (no retraining needed)."""
import json, os, copy, time
import numpy as np
import torch
import torch.nn as nn
import torchvision

from compress import fuse_bn, decompose_convs, collect_ks

OUT = os.path.dirname(os.path.abspath(__file__))
TAUS = [0.70, 0.80, 0.90]

def resnet18_cifar():
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
    return m

def mobilenetv2_cifar():
    m = torchvision.models.mobilenet_v2(num_classes=10)
    m.features[0][0].stride = (1, 1)
    return m

MODELS = {"resnet18": resnet18_cifar, "mobilenetv2": mobilenetv2_cifar}

def conv_flops(conv, oh, ow):
    Cout, Cin, H, W = conv.weight.shape
    g = conv.groups
    cin_g = Cin // g
    return oh * ow * Cout * cin_g * H * W * 2

def count_flops(model, in_shape=(1, 3, 32, 32)):
    """Hook-based FLOP counting at eval, input 3x32x32."""
    flops = {"total": 0}
    hooks = []
    def mk_hook(mod):
        def hook(m, inp, out):
            if isinstance(m, nn.Conv2d):
                oh, ow = out.shape[2], out.shape[3]
                flops["total"] += conv_flops(m, oh, ow)
            elif isinstance(m, nn.Linear):
                flops["total"] += m.in_features * m.out_features * 2 * out.shape[0]
        return hook
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(mk_hook(m)))
    model.eval()
    with torch.no_grad():
        model(torch.randn(*in_shape))
    for h in hooks: h.remove()
    return flops["total"]

def batched_latency(model, batch, n_warm=15, n_run=50):
    m = copy.deepcopy(model).cpu().eval()
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    x = torch.randn(batch, 3, 32, 32)
    with torch.no_grad():
        for _ in range(n_warm): m(x)
        ts = []
        for _ in range(n_run):
            t0 = time.perf_counter(); m(x)
            ts.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(prev_threads)          # restore: single-thread setting must not leak
    per_item = float(np.mean(ts)) / batch
    return per_item

def build(name, tau=None):
    m = MODELS[name]()
    m.load_state_dict(torch.load(os.path.join(OUT, f"{name}_best.pth"), map_location="cpu", weights_only=True))
    m = fuse_bn(m)
    if tau is not None:
        m = decompose_convs(m, tau)
    return m

if __name__ == "__main__":
    res = {}
    for name in MODELS:
        base = build(name)
        entry = {"flops_base": count_flops(base),
                 "lat_b1": batched_latency(base, 1),
                 "lat_b8": batched_latency(base, 8),
                 "lat_b32": batched_latency(base, 32)}
        for tau in TAUS:
            d = build(name, tau)
            entry[f"tau{tau}"] = {"flops": count_flops(d),
                                  "k_mean": float(np.mean(collect_ks(d))),
                                  "lat_b1": batched_latency(d, 1),
                                  "lat_b8": batched_latency(d, 8),
                                  "lat_b32": batched_latency(d, 32)}
            print(f"[{name}] tau={tau} flops={entry[f'tau{tau}']['flops']/1e6:.0f}M "
                  f"(x{entry['flops_base']/entry[f'tau{tau}']['flops']:.2f}) "
                  f"b8={entry[f'tau{tau}']['lat_b8']:.3f}ms/item vs base {entry['lat_b8']:.3f}", flush=True)
        res[name] = entry
    with open(os.path.join(OUT, "supplement.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("SUPPLEMENT DONE", flush=True)
