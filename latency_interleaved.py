# -*- coding: utf-8 -*-
"""Interleaved latency benchmark: all methods measured round-robin inside the
SAME time window, so machine-load drift affects every method equally.
Batch=8, single-thread CPU, median of R interleaved rounds per method.
"""
import argparse, copy, json, os, statistics, time
import torch

from baselines import get_loaders, make_model
from compress import fuse_bn, decompose_convs
from supplement import mobilenetv2_cifar

OUT = os.path.dirname(os.path.abspath(__file__))
LINE = "-" * 60

def build_all(taus, ratios, with_int8, in_size=32):
    """Returns {name: eval-mode CPU model}."""
    models = {}
    base = make_model("cifar10")
    fused = fuse_bn(copy.deepcopy(base)).eval()
    models["baseline_fp32"] = fused
    for tau in taus:
        m = decompose_convs(fuse_bn(copy.deepcopy(base)), tau)
        ck = os.path.join(OUT, f"svd_tau{tau}_kd.pth")
        if os.path.exists(ck):
            m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
        models[f"svd_tau{tau}"] = m.cpu().eval()
    for r in ratios:
        ck = os.path.join(OUT, f"depgraph_r{r}_kd.pth")
        if not os.path.exists(ck):
            print(f"skip depgraph r={r}: no KD checkpoint yet"); continue
        # rebuild pruned architecture deterministically, then load KD weights
        import torch_pruning as tp
        m = copy.deepcopy(base).cuda()
        ex = torch.randn(1, 3, in_size, in_size).cuda()
        tp.pruner.MagnitudePruner(model=m, example_inputs=ex,
                                  importance=tp.importance.MagnitudeImportance(p=1),
                                  iterative_steps=1, pruning_ratio=r,
                                  ignored_layers=[m.fc], global_pruning=True).step()
        m = m.cpu()
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
        m = fuse_bn(m)
        models[f"depgraph_r{r}"] = m.cpu().eval()
    if with_int8:
        from torch.ao.quantization import quantize_fx
        qm = torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping("fbgemm")
        tl, vl, _, _ = get_loaders("cifar10", bs=64, workers=0)
        prepared = quantize_fx.prepare_fx(copy.deepcopy(fused), qm,
                                          example_inputs=(torch.randn(1, 3, in_size, in_size),))
        n = 0
        with torch.no_grad():
            for x, _ in tl:
                prepared(x); n += 1
                if n >= 20: break
        models["int8_static"] = quantize_fx.convert_fx(prepared).eval()
    return models

def interleaved_bench(models, batch=8, rounds=12, n_inner=6):
    """Round-robin: each round times every model n_inner times; median across rounds."""
    torch.set_num_threads(1)
    x = torch.randn(batch, 3, 32, 32)
    mlist = [(name, copy.deepcopy(m).cpu().eval()) for name, m in models.items()]
    # warmup all
    with torch.no_grad():
        for _, m in mlist:
            for _ in range(10): m(x)
    samples = {name: [] for name, _ in mlist}
    for r in range(rounds):
        for name, m in mlist:
            ts = []
            with torch.no_grad():
                for _ in range(n_inner):
                    t0 = time.perf_counter(); m(x); ts.append((time.perf_counter() - t0) * 1000)
            samples[name].append(min(ts))          # min per block = least-contaminated
        print(f"  round {r+1}/{rounds} done", flush=True)
    out = {}
    for name, ss in samples.items():
        med = statistics.median(ss)
        out[name] = {"median_ms": med, "min_ms": min(ss), "max_ms": max(ss), "all": [round(v, 3) for v in ss]}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="0.70,0.80,0.90")
    ap.add_argument("--ratios", default="0.30,0.50")
    ap.add_argument("--no_int8", action="store_true")
    args = ap.parse_args()

    models = build_all([float(t) for t in args.taus.split(",")],
                       [float(r) for r in args.ratios.split(",")],
                       not args.no_int8)
    print(LINE); print("models:", list(models.keys()), flush=True)
    res = interleaved_bench(models, batch=8)
    base = res["baseline_fp32"]["median_ms"]
    print(LINE)
    for k, v in res.items():
        print(f"{k:16s} median={v['median_ms']:7.2f}ms  speedup={base/v['median_ms']:.2f}x  "
              f"[{v['min_ms']:.2f}..{v['max_ms']:.2f}]")
    with open(os.path.join(OUT, "latency_interleaved.json"), "w", encoding="utf-8") as f:
        json.dump({"batch": 8, "baseline_median_ms": base, "results": res}, f, indent=2)
    print("LATENCY DONE")
