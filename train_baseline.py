# -*- coding: utf-8 -*-
"""Train CIFAR-10 baseline models: ResNet-18 (CIFAR variant) and MobileNetV2."""
import json, time, os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models import mobilenet_v2

OUT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

def resnet18_cifar(num_classes=10):
    """ResNet-18 adapted for 32x32 (3x3 stem, no maxpool) — standard CIFAR variant."""
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

def mobilenetv2_cifar(num_classes=10):
    m = mobilenet_v2(num_classes=num_classes)
    # adapt for 32x32: first conv stride 1 (torchvision MobileNetV2 has no maxpool)
    m.features[0][0].stride = (1, 1)
    return m

def get_loaders(bs=256, workers=4):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(mean, std)])
    te = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    root = os.path.join(OUT, "data")
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=te)
    tl = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True,
                                     num_workers=workers, pin_memory=True, persistent_workers=True)
    vl = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False,
                                     num_workers=workers, pin_memory=True)
    return tl, vl

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total

def train(model, name, epochs=20, lr=1e-3):
    tl, vl = get_loaders()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0); n += y.size(0)
        sched.step()
        acc = evaluate(model, vl)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(OUT, f"{name}_best.pth"))
        print(f"[{name}] epoch {ep}/{epochs} loss={loss_sum/n:.4f} test_acc={acc*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), os.path.join(OUT, f"{name}_final.pth"))
    return best

if __name__ == "__main__":
    results = {}
    for name, fn in [("resnet18", resnet18_cifar), ("mobilenetv2", mobilenetv2_cifar)]:
        torch.manual_seed(42)
        acc = train(fn(), name, epochs=20)
        results[name] = {"best_test_acc": acc}
        print(f"== {name} best acc: {acc*100:.2f}% ==", flush=True)
    with open(os.path.join(OUT, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("TRAINING DONE", flush=True)
