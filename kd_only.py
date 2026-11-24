# -*- coding: utf-8 -*-
"""Round-4 m9 control: KD polish applied to the UNCOMPRESSED fused baseline.
Answers 'is KD merely paying back compression damage?': if KD on the full
model gains ~nothing, the +1.8pp KD gain on compressed students is genuine
recovery, not debt repayment. Protocol identical to the pipeline's KD stage:
teacher = fused FP32 baseline, AdamW lr=5e-5, T=2, alpha=0.5, 3 epochs."""
import copy, json, os, torch

from baselines import get_loaders, DEVICE, OUT
from finetune_utils import evaluate
from kd_exp import kd_polish
from compress import fuse_bn
from train_baseline import resnet18_cifar

OUT_JSON = os.path.join(OUT, "kd_only.json")

tl, vl, in_size, nc = get_loaders("cifar10", workers=0)

base = resnet18_cifar(num_classes=10)
base.load_state_dict(torch.load(os.path.join(OUT, "resnet18_best.pth"),
                                map_location="cpu", weights_only=True))
teacher = fuse_bn(copy.deepcopy(base)).eval()
acc0 = evaluate(teacher, vl, DEVICE)
print(f"fused baseline (teacher): {acc0*100:.2f}%", flush=True)

student = copy.deepcopy(teacher)
student, acc_kd = kd_polish(student, teacher, tl, vl, epochs=3, lr=5e-5)
res = {"kd_only_baseline_acc": acc0, "kd_only_acc": acc_kd,
       "gain_pp": (acc_kd - acc0) * 100}
json.dump(res, open(OUT_JSON, "w"), indent=1)
print("KD-only:", res, flush=True)
