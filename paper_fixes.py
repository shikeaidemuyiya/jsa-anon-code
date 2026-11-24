# -*- coding: utf-8 -*-
"""按 JSA-peer-review.md 要求补算所有数字：
M2  对齐/未对齐/基线 三布局 DRAM 流量与跨行浪费消融
M5  Table 3 正确口径（与 results.json/Eq(5) 一致）
m6  真实 FC 偏置维度的差分 exp-Golomb 压缩
m7  FCQ 字节核算（码本K/索引/bits-per-weight）
m11 ImageNet-100 +FCQ 体积来源分解
S2a 密度峰值种子 vs k-means 种子（同 K）精度对照
M4  环境/版本信息
"""
import copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
from compress import (fuse_bn, decompose_convs, diff_encode_bias,
                      choose_K_adaptive, cluster_quantize)
R = {}

# ---------------------------------------------------------------- M4 environment
import torchvision, platform
R["env"] = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": np.__version__,
    "os": f"{platform.system()} {platform.release()}",
    "cpu": "Intel Core i9-12900 (16 cores: 8P+8E), 64 GB RAM",
    "threads": "torch.set_num_threads(1) for all latency runs",
    "seed": 42,
    "augmentation": "RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize (CIFAR-10)",
}

# ---------------------------------------------------------------- M2/M5 traffic ablation
LINE = 64
def lines_span_unaligned(byte_start, byte_end):
    return int((byte_end - 1) // LINE) - int(byte_start // LINE) + 1

def traffic_contiguous(channels_bytes):
    """Contiguous unaligned layout: channels stored back-to-back."""
    total = sum(channels_bytes)
    pos, lines = 0, 0
    for b in channels_bytes:
        lines += lines_span_unaligned(pos, pos + b)
        pos += b
    return {"traffic_bytes": lines * LINE, "touched_lines": lines,
            "raw_bytes": total, "straddle_waste_bytes": lines * LINE - total}

def traffic_aligned(channels_bytes):
    lines = sum(math.ceil(b / LINE) for b in channels_bytes)
    total = sum(channels_bytes)
    return {"traffic_bytes": lines * LINE, "touched_lines": lines,
            "raw_bytes": total, "straddle_waste_bytes": lines * LINE - total}

def analyze_layer(cout, cin, hw, ks):
    out = {}
    dense = [cin * hw * 4] * cout   # per-oc weight bytes, dense conv
    out["baseline_unaligned"] = traffic_contiguous(dense)
    out["baseline_aligned"] = traffic_aligned(dense)
    for k in ks:
        s3 = [cin * hw * 4] * k         # spatial stage: k output channels
        p1 = [k * 4] * cout             # pointwise stage: cout blocks of k floats
        out[f"k{k}_unaligned"] = (traffic_contiguous(s3), traffic_contiguous(p1))
        out[f"k{k}_aligned"] = (traffic_aligned(s3), traffic_aligned(p1))
        raw = sum(s3) + sum(p1)
        aligned = traffic_aligned(s3)["traffic_bytes"] + traffic_aligned(p1)["traffic_bytes"]
        unal = traffic_contiguous(s3)["traffic_bytes"] + traffic_contiguous(p1)["traffic_bytes"]
        base = traffic_contiguous(dense)["traffic_bytes"]
        out[f"k{k}_summary"] = {"raw_bytes": raw, "aligned_traffic": aligned,
                                "unaligned_traffic": unal, "baseline_traffic": base,
                                "red_base_vs_aligned": base / aligned,
                                "red_base_vs_unaligned": base / unal,
                                "align_overhead_vs_unaligned": unal / aligned}
    return out

R["layer_256x256x9"] = analyze_layer(256, 256, 9, [16, 32, 64, 96])
R["layer_512x512x9"] = analyze_layer(512, 512, 9, [16, 32, 64, 96])
R["layer_128x128x9"] = analyze_layer(128, 128, 9, [16, 32, 64, 96])
R["layer_64x64x9"] = analyze_layer(64, 64, 9, [16, 32, 64, 96])

# ---------------------------------------------------------------- LRU miss-rate sim (reuse regime)
from compress import simulate_cache_weight_access
def lru_sim(per_oc_bytes, num_oc, pixels, cache=32 * 1024):
    return simulate_cache_weight_access(per_oc_bytes, num_oc, pixels, cache_size=cache)
R["lru"] = {
    "baseline_dense_oc9216B": lru_sim(256 * 9 * 4, 256, 256),
    "decomposed_pw_k16_oc64B": lru_sim(16 * 4, 256, 256),
    "decomposed_spatial_k16_oc9216B": lru_sim(16 * 256 * 9 * 4, 16, 256),
    "decomposed_pw_k4_oc16B": lru_sim(4 * 4, 256, 256),
}

# ---------------------------------------------------------------- m6 bias coding on real dims
def fc_bias_stats(model, name):
    fc = model.fc if hasattr(model, "fc") else model.classifier[1]
    r = diff_encode_bias(fc.bias.detach().numpy())
    r["dim"] = int(fc.bias.numel()); r["layer"] = name
    return r

def resnet18_cifar(num_classes=10):
    import torch.nn as nn_
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn_.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn_.Identity()
    return m

m0 = resnet18_cifar(num_classes=10)
m0.load_state_dict(torch.load(os.path.join(OUT, "resnet18_best.pth"), map_location="cpu", weights_only=True))
m0f = fuse_bn(copy.deepcopy(m0))
R["bias_cifar"] = fc_bias_stats(m0f, "ResNet-18/CIFAR-10 fc (512->10)")

mi = torchvision.models.resnet18(num_classes=100)  # ImageNet-100 teacher (7x7 stem)
mi.load_state_dict(torch.load(os.path.join(OUT, "resnet18_imagenet100_best.pth"), map_location="cpu", weights_only=True))
mif = fuse_bn(copy.deepcopy(mi))
R["bias_in100"] = fc_bias_stats(mif, "ResNet-18/ImageNet-100 fc (512->100)")

# ---------------------------------------------------------------- m7 FCQ byte accounting
def fcq_accounting(W, cache_bytes=32 * 1024):
    W = W.detach()
    sel = choose_K_adaptive(W.numpy(), cache_bytes=cache_bytes)
    K = sel["K"]
    centers, a, counts = cluster_quantize(W.flatten().numpy(), K)
    idx_bits = 16
    idx_bytes = a.size * idx_bits // 8
    used = sel["used_clusters"]
    codebook_bytes = K * 4 + used * 2
    return {"K": K, "used_clusters": used, "coverage": round(sel["coverage"], 4),
            "fp32_bytes": W.numel() * 4, "index_bytes": idx_bytes,
            "codebook_bytes": codebook_bytes,
            "quant_bytes": idx_bytes + codebook_bytes,
            "bits_per_weight": (idx_bytes + codebook_bytes) * 8 / W.numel(),
            "compression": W.numel() * 4 / (idx_bytes + codebook_bytes),
            "dc_mult": sel["dc_mult"]}

R["fcq_cifar_fc"] = fcq_accounting(m0f.fc.weight)
R["fcq_in100_fc"] = fcq_accounting(mif.fc.weight)

# full-model FCQ size delta (ImageNet-100 tau0.9+KD), mirrors Table 4
p_before = sum(p.numel() for p in mif.parameters())
R["in100_size"] = {"params": p_before, "fp32_MB": p_before * 4 / 1e6}

# ---------------------------------------------------------------- S2a density-peak vs k-means seeding
from compress import cluster_quantize
def seed_compare(W, K):
    W = W.detach()
    flat = W.flatten().numpy().astype(np.float64)
    _, a_dp, c_dp = cluster_quantize(flat, K)          # DP seeds + kmeans refine
    rng = np.random.default_rng(42)
    km_idx = rng.choice(flat.size, K, replace=False)   # pure k-means (random init)
    try:
        from scipy.cluster.vq import kmeans2
        c_km, a_km = kmeans2(flat, flat[km_idx], minit="matrix", iter=15, seed=42)
    except Exception:
        # numpy fallback Lloyd's
        c_km = flat[km_idx].copy()
        for _ in range(15):
            d = (flat[:, None] - c_km[None, :]) ** 2
            a_km = d.argmin(1)
            for j in range(K):
                m = flat[a_km == j]
                if m.size: c_km[j] = m.mean()
    err_dp = float(np.mean((c_dp[a_dp].reshape(W.shape) - W.numpy()) ** 2))
    err_km = float(np.mean((c_km[a_km].reshape(W.shape) - W.numpy()) ** 2))
    return {"K": K, "dp_mse": err_dp, "kmeans_mse": err_km}
R["seed_compare_cifar"] = seed_compare(m0f.fc.weight, R["fcq_cifar_fc"]["K"])
R["seed_compare_in100"] = seed_compare(mif.fc.weight, R["fcq_in100_fc"]["K"])

# ---------------------------------------------------------------- S2a-2 accuracy: fc-only quantization, both seedings (1k val images)
import torchvision.transforms as T
import torchvision
te = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
val = torchvision.datasets.CIFAR10(os.path.join(OUT, "data"), train=False, transform=te)
vl = torch.utils.data.DataLoader(val, batch_size=256, shuffle=False, num_workers=0)

def quantize_fc_variant(model, mode, K):
    m = copy.deepcopy(model).eval()
    fc = m.fc
    W = fc.weight.detach()
    flat = W.flatten().numpy().astype(np.float64)
    if mode == "dp":
        centers, a, _ = cluster_quantize(flat, K)
    else:
        try:
            from scipy.cluster.vq import kmeans2
            rng = np.random.default_rng(42)
            c0 = flat[rng.choice(flat.size, K, replace=False)]
            centers, a = kmeans2(flat, c0, minit="matrix", iter=15, seed=42)
            centers, a = np.asarray(centers), np.asarray(a)
        except Exception:
            rng = np.random.default_rng(42)
            centers = flat[rng.choice(flat.size, K, replace=False)].copy()
            for _ in range(15):
                d = (flat[:, None] - centers[None, :]) ** 2
                a = d.argmin(1)
                for j in range(K):
                    mseg = flat[a == j]
                    if mseg.size: centers[j] = mseg.mean()
    new = copy.deepcopy(fc)
    with torch.no_grad():
        new.weight.copy_(torch.from_numpy(centers[a].reshape(W.shape).astype(np.float32)))
    m.fc = new
    return m

@torch.no_grad()
def ev(m, n=2000):
    m.eval().cpu(); c = t = 0
    for x, y in vl:
        c += (m(x).argmax(1) == y).sum().item(); t += y.size(0)
        if t >= n: break
    return c / t

Kc = R["fcq_cifar_fc"]["K"]
R["acc_fc_only"] = {
    "fp32_fused_1k": round(ev(copy.deepcopy(m0f)), 4),
    "dp_seed": round(ev(quantize_fc_variant(m0f, "dp", Kc)), 4),
    "kmeans_seed": round(ev(quantize_fc_variant(m0f, "kmeans", Kc)), 4),
    "K": Kc,
}

json.dump(R, open(os.path.join(OUT, "paper_fixes.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float)[:4000])
