# -*- coding: utf-8 -*-
"""诊断：ImageNet-100 上分解后模型的 acc_before_ft 与各层秩"""
import copy, os, torch, torchvision
from datasets import load_from_disk
import torchvision.transforms as T
from compress import fuse_bn, decompose_convs, collect_ks, LowRankConv

OUT = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class HFWrap(torch.utils.data.Dataset):
    def __init__(self, ds, tf): self.ds, self.tf = ds, tf
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[int(i)]; return self.tf(r["image"]), r["label"]

@torch.no_grad()
def ev(m, vl, n=1000):
    m.eval().to(DEV)
    c = t = 0
    for x, y in vl:
        x, y = x.to(DEV), y.to(DEV)
        pred = m(x).argmax(1)
        c += (pred == y).sum().item(); t += y.size(0)
        if t >= n: break
    return c / t

te = T.Compose([T.Grayscale(3), T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
d = load_from_disk(os.path.join(OUT, "imagenet100"))
vl = torch.utils.data.DataLoader(HFWrap(d["validation"], te), batch_size=64, num_workers=0)

m0 = torchvision.models.resnet18(num_classes=100)
m0.load_state_dict(torch.load(os.path.join(OUT, "resnet18_imagenet100_best.pth"), map_location="cpu", weights_only=True))
print(f"raw best ckpt acc={ev(copy.deepcopy(m0), vl)*100:.2f}%", flush=True)

f = fuse_bn(copy.deepcopy(m0))
print(f"fused acc={ev(copy.deepcopy(f), vl)*100:.2f}%", flush=True)
for tau in (0.7, 0.8, 0.9):
    dm = decompose_convs(copy.deepcopy(f), tau)
    ks = collect_ks(dm)
    a = ev(copy.deepcopy(dm), vl)
    print(f"tau={tau} acc_before_ft={a*100:.2f}% k_mean={sum(ks)/len(ks):.1f} ks={ks}", flush=True)
