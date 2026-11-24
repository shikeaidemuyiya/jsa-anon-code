# -*- coding: utf-8 -*-
import copy, os, json, torch, torchvision
from compress import fuse_bn, decompose_convs, collect_ks
from baselines import count_flops

OUT = os.path.dirname(os.path.abspath(__file__))
m0 = torchvision.models.resnet18(num_classes=100)
m0.load_state_dict(torch.load(os.path.join(OUT, "resnet18_imagenet100_best.pth"), map_location="cpu", weights_only=True))
f = fuse_bn(copy.deepcopy(m0))
base_f = count_flops(f, in_shape=(1, 3, 224, 224))
base_p = sum(p.numel() for p in f.parameters())
print(f"baseline flops={base_f/1e9:.3f}G params={base_p/1e6:.2f}M")
for tau in (0.7, 0.8, 0.9, 0.95, 0.98, 0.99):
    d = decompose_convs(copy.deepcopy(f), tau)
    fl = count_flops(d, in_shape=(1, 3, 224, 224))
    p = sum(x.numel() for x in d.parameters())
    ks = collect_ks(d)
    print(f"tau={tau}: flops={fl/1e9:.3f}G ({base_f/fl:.2f}x) params={p/1e6:.2f}M ({base_p/p:.2f}x) k_mean={sum(ks)/len(ks):.1f}")
print("\n--- CIFAR reference ---")
r = json.load(open("results.json"))
for tau, v in r.get("svd", {}).items():
    print(tau, "acc_before_ft=%.2f%%" % (v.get("acc_before_ft", -1)*100), "params=%d" % v["params"], "flops=%.3fG" % (v["flops"]/1e9))
