# -*- coding: utf-8 -*-
"""ImageNet-100 CPU single-thread interleaved latency (paper-consistent protocol).

Same protocol as latency_interleaved.py: round-robin inside one time window,
batch=8, single thread, median over interleaved rounds. Input 224x224.
Model = skip-stem SVD tau=0.9 + KD weights, vs fused baseline.
"""
import copy, json, os, statistics, time
import torch

from run_in100_v3 import decompose_skip_stem, teacher_model
from compress import fuse_bn
from run_experiments import quantize_all_fc

OUT = os.path.dirname(os.path.abspath(__file__))


def build_models():
    base = teacher_model()
    fused = fuse_bn(copy.deepcopy(base)).eval().cpu()
    models = {"baseline_fp32": fused}

    m = decompose_skip_stem(fuse_bn(copy.deepcopy(base)), 0.90)
    ck = os.path.join(OUT, "in100_tau0.9_kd.pt")
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["best_state"])
    models["ours_tau0.9_kd"] = m.cpu().eval()

    mq, info = quantize_all_fc(copy.deepcopy(m))
    models["ours_tau0.9_kd_fcq"] = mq.cpu().eval()
    return models


def interleaved_bench(models, batch=8, rounds=10, n_inner=4):
    torch.set_num_threads(1)
    x = torch.randn(batch, 3, 224, 224)
    mlist = [(name, copy.deepcopy(m).cpu().eval()) for name, m in models.items()]
    with torch.no_grad():
        for _, m in mlist:
            for _ in range(2):
                m(x)                                   # warmup
    samples = {name: [] for name, _ in mlist}
    for r in range(rounds):
        for name, m in mlist:
            with torch.no_grad():
                inner = []
                for _ in range(n_inner):
                    t0 = time.perf_counter(); m(x)
                    inner.append((time.perf_counter() - t0) * 1000)
            samples[name].append(min(inner))
        print(f"  round {r+1}/{rounds} done", flush=True)
    return {n: {"median_ms": statistics.median(ss), "all": [round(v, 2) for v in ss]}
            for n, ss in samples.items()}


if __name__ == "__main__":
    models = build_models()
    print("models:", list(models.keys()), flush=True)
    res = interleaved_bench(models)
    base = res["baseline_fp32"]["median_ms"]
    print("\n=== ImageNet-100 CPU single-thread interleaved (batch=8, 224x224) ===")
    for k, v in res.items():
        print(f"{k:20s} median={v['median_ms']:8.2f}ms  speedup={base/v['median_ms']:.2f}x")
    with open(os.path.join(OUT, "latency_in100_cpu.json"), "w", encoding="utf-8") as f:
        json.dump({"batch": 8, "input": "224x224", "baseline_median_ms": base,
                   "results": res}, f, indent=2)
    print("LATENCY IN100 DONE")
