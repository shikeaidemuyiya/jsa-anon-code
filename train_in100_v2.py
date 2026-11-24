# -*- coding: utf-8 -*-
"""Self-contained ImageNet-100 fine-tuning of pretrained ResNet-18 (AMP, batch 96).

Separate from baselines.py on purpose: that module is imported by running jobs and
must not be edited mid-run on Windows (spawned workers re-import it).
"""
import os, time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from datasets import load_from_disk

OUT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class HFWrap(torch.utils.data.Dataset):
    def __init__(self, ds, tf): self.ds = ds; self.tf = tf
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[int(i)]
        return self.tf(r["image"]), r["label"]


def loaders(bs=64):
    tr = T.Compose([T.Grayscale(3), T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(MEAN, STD)])
    te = T.Compose([T.Grayscale(3), T.Resize(256), T.CenterCrop(224),
                    T.ToTensor(), T.Normalize(MEAN, STD)])
    d = load_from_disk(os.path.join(OUT, "imagenet100"))
    tl = torch.utils.data.DataLoader(HFWrap(d["train"], tr), batch_size=bs, shuffle=True,
                                     num_workers=3, pin_memory=True, drop_last=True)
    vl = torch.utils.data.DataLoader(HFWrap(d["validation"], te), batch_size=128, shuffle=False,
                                     num_workers=4, pin_memory=True)
    return tl, vl


@torch.no_grad()
def evaluate(model, vl):
    model.eval()
    correct = total = 0
    for x, y in vl:
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        correct += (out.float().argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total


if __name__ == "__main__":
    tl, vl = loaders()
    m = torchvision.models.resnet18(num_classes=100)
    sd = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
    sd["fc.weight"] = m.fc.weight.data; sd["fc.bias"] = m.fc.bias.data
    m.load_state_dict(sd)
    m = m.to(DEVICE)

    epochs = 6
    opt = torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    best = 0.0
    for ep in range(epochs):
        t0 = time.time(); m.train()
        for i, (x, y) in enumerate(tl):
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(m(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        acc = evaluate(m, vl)
        print(f"ep{ep+1}/{epochs} acc={acc*100:.2f}% ({time.time()-t0:.0f}s) "
              f"lr={sched.get_last_lr()[0]:.4f}", flush=True)
        if acc > best:
            best = acc
            torch.save({k: v.float() for k, v in m.state_dict().items()},
                       os.path.join(OUT, "resnet18_imagenet100_best.pth"))
    print(f"BEST {best*100:.2f}", flush=True)
    print("TRAIN DONE", flush=True)
