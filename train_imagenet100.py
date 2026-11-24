# -*- coding: utf-8 -*-
"""Fine-tune pretrained ResNet-18 on ImageNet-100 to create the baseline checkpoint."""
import json, os, time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from baselines import get_loaders, DEVICE, OUT, measure
from compress import fuse_bn

if __name__ == "__main__":
    tl, vl, in_size, nc = get_loaders("imagenet100")
    m = torchvision.models.resnet18(num_classes=100)
    sd = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
    sd["fc.weight"] = m.fc.weight.data; sd["fc.bias"] = m.fc.bias.data
    m.load_state_dict(sd)
    m = m.to(DEVICE)

    epochs = 8
    opt = torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        t0 = time.time()
        m.train()
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(m(x), y).backward()
            opt.step(); sched.step()
        # fused-model accuracy (that's the model we compress later)
        m_cpu = fuse_bn(m.cpu())
        acc = measure(m_cpu, vl, 224)
        print(f"ep{ep+1}/{epochs} acc={acc['acc']*100:.2f}% flops={acc['flops']/1e9:.2f}G "
              f"({time.time()-t0:.0f}s)", flush=True)
        if acc["acc"] > best:
            best = acc["acc"]
            torch.save(m.state_dict(), os.path.join(OUT, "resnet18_imagenet100_best.pth"))
        m = m.to(DEVICE)
    print(f"BEST {best*100:.2f}", flush=True)
    print("TRAIN DONE", flush=True)
