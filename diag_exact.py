# -*- coding: utf-8 -*-
"""无损性检验：tau=1.0 时分解应等价于原卷积；并统计权重谱的有效秩"""
import copy, os, torch, torchvision
from compress import fuse_bn, decompose_convs, LowRankConv, rank_for_energy

OUT = os.path.dirname(os.path.abspath(__file__))
m0 = torchvision.models.resnet18(num_classes=100)
m0.load_state_dict(torch.load(os.path.join(OUT, "resnet18_imagenet100_best.pth"), map_location="cpu", weights_only=True))
f = fuse_bn(copy.deepcopy(m0)).eval()

# 单层无损性
c = f.layer1[0].conv1
x = torch.randn(2, 64, 56, 56)
with torch.no_grad():
    y0 = c(x)
    for tau in (0.99, 1.0):
        lr = LowRankConv(c, tau)
        y1 = lr(x)
        print(f"tau={tau} k={lr.k}/{min(lr.s3.out_channels, 64*9)} maxdiff={torch.abs(y0-y1).max().item():.3e} "
              f"relerr={(torch.abs(y0-y1).max()/y0.abs().max()).item():.3e}")

# 全模型无损性
d = decompose_convs(copy.deepcopy(f), 1.0).eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    a, b = f(x), d(x)
print(f"full model tau=1.0 maxdiff={(a-b).abs().max().item():.4e} argmax_same={(a.argmax(1)==b.argmax(1)).item()}")

# 谱有效秩：CIFAR 模型 vs ImageNet 模型同名层
import json
cf = torchvision.models.resnet18(num_classes=10)
cf.load_state_dict(torch.load(os.path.join(OUT, "resnet18_best.pth"), map_location="cpu", weights_only=True))
cfg = fuse_bn(cf).eval()
print("\nlayer                 | CIFAR effrank/rank tau@0.9 | ImageNet effrank/rank tau@0.9")
for name in ["layer1.0.conv1", "layer2.0.conv1", "layer3.1.conv2", "layer4.1.conv2"]:
    mods = dict(cfg.named_modules())
    modi = dict(f.named_modules())
    out = [name]
    for m in (mods[name], modi[name]):
        W = m.weight.data.reshape(m.weight.shape[0], -1)
        S = torch.linalg.svdvals(W)
        e = (S**2); pr = (e.sum()**2) / (e**2).sum()   # participation ratio
        out += [f"{pr.item():6.1f}/{len(S):4d} k90={rank_for_energy(S,0.9):3d}"]
    print(" | ".join(out))
