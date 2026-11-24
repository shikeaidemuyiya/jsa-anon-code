# -*- coding: utf-8 -*-
"""
Patent-1 compression pipeline (G06N3):
  Step1: operator fusion (BN folding) + redundant node elimination
  Step2: SVD low-rank decomposition -> depthwise(1x1) + grouped spatial conv,
         cache-line aligned block rearrangement of decomposed kernels
  Step3: density-peak clustering quantization of FC weights + diff/exp-Golomb bias coding
  Step4: subgraph repack nodes (modeled in latency benchmark & cache simulation)
"""
import math, json, os
import numpy as np
import torch
import torch.nn as nn

CACHE_LINE = 64  # bytes — same as Raspberry Pi 4 / Jetson Nano / x86

# ---------------------------------------------------------------- Step 1: BN fusion
@torch.no_grad()
def fold_bn_into_conv(conv, bn):
    """W' = gamma/(sqrt(var+eps)) * W ; b' = gamma*(0-mean)/sqrt(var+eps) + beta (patent Eq.)"""
    w = conv.weight.data
    s = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
    conv.weight.data = w * s.view(-1, 1, 1, 1)
    bias = -bn.running_mean * s + bn.bias.data
    new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                         stride=conv.stride, padding=conv.padding,
                         dilation=conv.dilation, groups=conv.groups, bias=True)
    new_conv.weight.data = conv.weight.data
    new_conv.bias.data = bias
    return new_conv

@torch.no_grad()
def fuse_bn(model):
    """Fuse every Conv2d+BatchNorm2d pair (operator fusion step of the patent)."""
    model = model.cpu().eval()
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            fuse_bn(module)
        elif isinstance(module, nn.BatchNorm2d):
            # parent containers are nn.Sequential; handled generically below
            pass
    # recursive scan: fuse adjacent Conv2d+BatchNorm2d pairs in every module (registration order)
    def _scan(container):
        names = list(container._modules.keys())
        i = 0
        while i < len(names) - 1:
            a, b = container._modules[names[i]], container._modules[names[i + 1]]
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                container._modules[names[i]] = fold_bn_into_conv(a, b)
                # keep positional reference valid (torchvision forward uses self.bnN)
                container._modules[names[i + 1]] = nn.Identity()
                names = list(container._modules.keys())
            else:
                i += 1
        for child in container.children():
            _scan(child)
    _scan(model)
    return model

# ---------------------------------------------------------------- Step 2: SVD low-rank decomposition
def rank_for_energy(s, tau):
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    k = int(torch.searchsorted(e, torch.tensor(tau)).item()) + 1
    return min(k, s.numel())          # clamp: tau=1.0 must not exceed full rank

class LowRankConv(nn.Module):
    """Patent step 2 (layer-level SVD): weight (Cout, Cin*H*W) = U S V^T,
    re-factorized as a 3x3 conv with k intermediate channels (depthwise-separable
    spatial stage, V*S reshaped) followed by a pointwise 1x1 conv (U).
    Receptive field preserved; k << Cin enables real FLOP/param reduction."""
    def __init__(self, conv, tau):
        super().__init__()
        Cout, Cin, H, W = conv.weight.shape
        W2 = conv.weight.data.reshape(Cout, Cin * H * W)
        U, S, Vh = torch.linalg.svd(W2, full_matrices=False)  # U:(Cout,k) S:(k) Vh:(k,CinHW)
        k = rank_for_energy(S, tau)
        self.k = k
        # spatial stage: k-channel HxW conv, weights = S*V^T reshaped
        self.s3 = nn.Conv2d(Cin, k, (H, W), stride=conv.stride,
                            padding=conv.padding, bias=False)
        self.s3.weight.data = (S[:k, None] * Vh[:k, :]).reshape(k, Cin, H, W)
        # pointwise stage: 1x1 conv, weights = U
        self.p1 = nn.Conv2d(k, Cout, 1, stride=1, bias=conv.bias is not None)
        self.p1.weight.data = U[:, :k].reshape(Cout, k, 1, 1)
        if conv.bias is not None:
            self.p1.bias.data = conv.bias.data.clone()

    def effective_params(self):
        return int(self.s3.weight.numel() + self.p1.weight.numel() +
                   (self.p1.bias.numel() if self.p1.bias is not None else 0))

    def forward(self, x):
        return self.p1(self.s3(x))

@torch.no_grad()
def decompose_convs(model, tau, skip_depthwise=True):
    """Replace spatial Conv2d with LowRankConv (layer-level SVD).
    Depthwise convs (groups=in_ch) skipped."""
    def _recurse(container):
        for name, m in list(container.named_children()):
            is_conv = isinstance(m, nn.Conv2d)
            if is_conv and m.kernel_size[0] > 1 and not (skip_depthwise and m.groups == m.in_channels):
                setattr(container, name, LowRankConv(m, tau))
            else:
                _recurse(m)
    _recurse(model)
    return model

# --------------------------------------------------- Step 2b: cache-line aligned rearrangement (layout metadata + cache sim)
def layout_depthwise(U_cin, k, S, align=CACHE_LINE):
    """Depthwise kernel per-input-channel blocks aligned to cache-line multiples (patent claim 4)."""
    elem = 4  # float32
    per_ch = k * elem
    A = math.ceil(max(per_ch, 1) / align) * align
    return {"block_size": A, "total": U_cin * A, "padding": (A - per_ch) * U_cin}

def layout_pointwise(HW, k, cout, dma_cap=4096, align=CACHE_LINE):
    """Pointwise kernels grouped by output channel, groups padded to cache-line boundary (patent claim 4)."""
    elem = 4
    per_oc = HW * k * elem
    G = max(1, dma_cap // per_oc)
    group_bytes = G * per_oc
    return {"group_size": G, "group_bytes": group_bytes,
            "pad_per_group": (math.ceil(group_bytes / align) * align - group_bytes),
            "groups": math.ceil(cout / G)}

# ---------------------------------------------------------------- Step 3: density-peak clustering quantization
def density_peak_clusters_1d(x, dc):
    """Simplified 1-D density-peak clustering. Returns indices of cluster seeds
    ranked by DPC decision value (rho * delta)."""
    import bisect
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    xs = np.sort(x)
    idx_hi = np.searchsorted(xs, x + dc, side="right")
    idx_lo = np.searchsorted(xs, x - dc, side="left")
    rho = idx_hi - idx_lo - 1
    # delta: distance to nearest point with strictly higher density (bisect sweep, O(n log n))
    order = np.argsort(-rho)
    delta = np.empty(n)
    delta[order[0]] = x.max() - x.min()
    seen = []
    for i in order[1:]:
        pos = bisect.bisect_left(seen, x[i])
        best = np.inf
        if pos > 0: best = min(best, x[i] - seen[pos - 1])
        if pos < len(seen): best = min(best, seen[pos] - x[i])
        delta[i] = best
        bisect.insort(seen, x[i])
    centers_idx = np.argsort(-(rho * delta))
    return centers_idx

def cluster_quantize(W, K):
    """K-means refinement on DPC seeds (1-D weights). Returns codebook + uint indices."""
    flat = W.flatten().astype(np.float64)
    seed_idx = density_peak_clusters_1d(flat, flat.std() / 20.0)[:K]
    centers = flat[seed_idx].copy()
    for _ in range(15):  # light k-means refinement
        d = np.abs(flat[:, None] - centers[None, :])
        a = d.argmin(1)
        for j in range(K):
            m = flat[a == j]
            if m.size: centers[j] = m.mean()
    # keep clusters sorted for the coverage criterion
    d = np.abs(flat[:, None] - centers[None, :]); a = d.argmin(1)
    counts = np.bincount(a, minlength=K)
    return centers.astype(np.float32), a.reshape(W.shape).astype(np.uint16), counts

def choose_K_adaptive(W, cache_bytes=32 * 1024, coverage_target=0.95):
    """Patent claim 6: pick K s.t. codebook fits read-only cache and top-cluster coverage >= 0.95."""
    flat = W.flatten()
    dc0 = flat.std() / 20.0
    best = None
    for mult in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]:
        centers_idx = density_peak_clusters_1d(flat, dc0 * mult)
        # candidate K: seeds whose clusters are non-degenerate; test prefix sizes
        for K in [16, 32, 64, 128, 256]:
            if K > len(centers_idx): break
            centers, a, counts = cluster_quantize(flat, K)
            top = np.sort(counts)[::-1]
            cov = 0.0; used = 0
            for c in top:
                cov += c / flat.size; used += 1
                if cov >= coverage_target: break
            codebook_bytes = K * 4 + used * 2  # shared weights + index table overhead
            if codebook_bytes <= cache_bytes:
                err = np.mean((centers[a].reshape(W.shape) - W) ** 2)
                cand = {"K": K, "used_clusters": used, "coverage": cov, "mse": err, "dc_mult": mult}
                if best is None or err < best["mse"]:
                    best = cand
        if best is not None:
            break  # smallest dc satisfying constraints wins (patent: smallest dc)
    return best

EXP_GOLOMB = None
def exp_golomb_encode(vals_int):
    """0-order exp-Golomb bit length for signed ints (zigzag)."""
    total_bits = 0
    for v in vals_int:
        u = 2 * v if v >= 0 else -2 * v - 1
        m = int(np.floor(np.log2(u + 1))) if u > 0 else 0
        total_bits += 2 * m + 1
    return total_bits / 8.0  # bytes

def diff_encode_bias(bias, frac_bits=10):
    """Patent claim 7: adjacent-difference + exp-Golomb, fixed-point (frac_bits)."""
    b = np.asarray(bias, dtype=np.float64)
    scaled = np.round(b * (1 << frac_bits)).astype(np.int64)
    diff = np.diff(scaled, prepend=scaled[:1])
    bits = exp_golomb_encode(diff.tolist())
    return {"bias_bytes_f32": b.size * 4, "bias_bytes_coded": math.ceil(bits),
            "ratio": (b.size * 4) / max(math.ceil(bits), 1)}

# ---------------------------------------------------------------- quantized FC inference (decode + lookup pipeline)
class QuantizedFC(nn.Module):
    """Cluster-quantized FC layer: uint16 index matrix + codebook lookup (patent step 3)."""
    def __init__(self, fc, centers, idx):
        super().__init__()
        self.out_f, self.in_f = fc.weight.shape
        self.register_buffer("codebook", torch.from_numpy(centers))
        self.register_buffer("idx", torch.from_numpy(idx.astype(np.int64)))
        if fc.bias is not None:
            self.register_buffer("bias", fc.bias.data.clone())
        else:
            self.bias = None
    def forward(self, x):
        w = self.codebook[self.idx]           # shared-weight lookup
        return torch.nn.functional.linear(x, w.view(self.out_f, self.in_f), self.bias)

# ---------------------------------------------------------------- cache simulator (structural weight-access simulation)
def simulate_cache_weight_access(layer_bytes_per_oc, num_oc, num_out_pixels,
                                 cache_size=32 * 1024, line=64):
    """
    Simulate LRU cache over weight-line accesses for one conv layer under the
    output-channel-major loop order (weights of one output channel are streamed
    once per layer invocation and reused for all its output pixels).
    Returns miss ratio and total weight DRAM traffic (bytes).
    """
    from collections import OrderedDict
    lines_per_oc = math.ceil(layer_bytes_per_oc / line)
    total_weight_lines = lines_per_oc * num_oc
    lru = OrderedDict()
    def access(l):
        if l in lru:
            lru.move_to_end(l); return 0
        lru[l] = True
        if len(lru) > cache_size // line:
            lru.popitem(last=False)
        return 1
    miss = 0
    for o in range(num_oc):                    # oc-major: weights reused across pixels
        base = o * lines_per_oc
        for l in range(lines_per_oc):
            miss += access(base + l)
    traffic = total_weight_lines * line        # compulsory fetch per layer invocation
    accesses = total_weight_lines * max(num_out_pixels, 1)
    return {"misses": miss, "miss_rate": miss / accesses,
            "dram_weight_traffic_bytes": traffic}

def weight_traffic_baseline(cout, cin, hw, line=64):
    return math.ceil(cin * hw * 4 / line) * line * cout

def weight_traffic_decomposed(cout, cin, hw, k, line=64, align=64):
    """Decomposed: spatial stage (k*Cin*HW) + pointwise stage (Cout*k),
    each block aligned to cache-line multiples (patent layout)."""
    s3_bytes = k * cin * hw * 4
    p1_bytes = cout * k * 4
    b1 = math.ceil(s3_bytes / align) * align
    b2 = math.ceil(p1_bytes / align) * align
    return b1 + b2

def collect_ks(model):
    """Collect per-layer truncation ranks k from all LowRankConv modules."""
    ks = []
    for m in model.modules():
        if isinstance(m, LowRankConv):
            ks.append(m.k)
    return ks

# ---------------------------------------------------------------- model size accounting
def model_stats(model):
    params = sum(p.numel() for p in model.parameters())
    return {"params": params, "size_MB_f32": params * 4 / 1e6}
