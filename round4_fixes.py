# -*- coding: utf-8 -*-
"""Round-4 fixes: three staging-transaction accounting models (A/B/C) per the
reviewer's companion note, plus per-layer aggregation over the 17 decomposed
ResNet-18 pointwise stages + MobileNetV2 stem, no-growth cap demo, and a
parametric timing example. Pure algebra, no torch.

Models (L = 64 B cache line, N channels, B = 4k bytes per channel block):
  A  per-channel independent transactions (no partial-line reuse across channels)
  B  line-streaming burst (each line fetched once, shared lines reused)
  C  worst-case single-channel amplification (order-agnostic)
"""
import json, math

L = 64
N_REF = 256  # uniform reference width used by Table 3


def offsets_packed(B, N):
    return [(i * B) % L for i in range(N)]


def A_packed(B, N):
    lines = sum(math.ceil((o + B) / L) for o in offsets_packed(B, N))
    return {"lines": lines, "overfetch": lines * L - N * B}


def A_aligned(B, N):
    n = math.ceil(B / L)
    return {"lines": N * n, "overfetch": N * (n * L - B)}


def B_packed(B, N):
    span = N * B
    lines = math.ceil(span / L)
    return {"lines": lines, "overfetch": lines * L - span}


def B_aligned(B, N):
    # fixed stride: neighbouring channels never share a line -> same as A
    return A_aligned(B, N)


def C_models(B, N):
    worst_p = max(math.ceil((o + B) / L) for o in offsets_packed(B, N))
    n = math.ceil(B / L)
    return {"packed_amp": worst_p * L / B, "aligned_amp": n * L / B}


def all_models(k, N):
    B = 4 * k
    return {
        "k": k, "B": B, "N": N,
        "A_packed": A_packed(B, N), "A_aligned": A_aligned(B, N),
        "B_packed": B_packed(B, N), "B_aligned": B_aligned(B, N),
        "C": C_models(B, N),
        "B_mod_64": B % 64,
    }


# ---------- 1. validation against the reviewer's companion table ----------
VALIDATION = {}
for k in (4, 13, 16, 32, 82, 96):
    r = all_models(k, N_REF)
    VALIDATION[k] = r
    print(f"k={k:3d} B={r['B']:3d}  A {r['A_packed']['lines']:5d}/{r['A_packed']['overfetch']:6d}"
          f" | {r['A_aligned']['lines']:5d}/{r['A_aligned']['overfetch']:6d}"
          f"  B {r['B_packed']['lines']:5d}/{r['B_packed']['overfetch']:6d}"
          f" | {r['B_aligned']['lines']:5d}/{r['B_aligned']['overfetch']:6d}"
          f"  C {r['C']['packed_amp']:.2f}x/{r['C']['aligned_amp']:.2f}x")

# expected values from the companion note (N=256)
EXPECTED = {
    4:  (256, 12288, 256, 12288, 64, 0, 4.00, 4.00),
    13: (448, 15360, 256, 3072, 208, 0, 2.46, 1.23),
    16: (256, 0, 256, 0, 256, 0, 1.00, 1.00),
    32: (512, 0, 512, 0, 512, 0, 1.00, 1.00),
    82: (1536, 14336, 1536, 14336, 1312, 0, 1.17, 1.17),
    96: (1536, 0, 1536, 0, 1536, 0, 1.00, 1.00),
}
ok = True
for k, (ap, ao, aa, aao, bp, bo, cp, ca) in EXPECTED.items():
    r = VALIDATION[k]
    checks = [
        r["A_packed"]["lines"] == ap, r["A_packed"]["overfetch"] == ao,
        r["A_aligned"]["lines"] == aa, r["A_aligned"]["overfetch"] == aao,
        r["B_packed"]["lines"] == bp, r["B_packed"]["overfetch"] == bo,
        abs(r["C"]["packed_amp"] - cp) < 0.01, abs(r["C"]["aligned_amp"] - ca) < 0.01,
    ]
    if not all(checks):
        ok = False
        print(f"VALIDATION MISMATCH at k={k}: {checks}")
print("validation:", "PASS" if ok else "FAIL")

# ---------- 2. per-layer aggregation (measured ranks from round3_fixes.json) ----------
round3 = json.load(open(r"./round3_fixes.json"))
AGG = {}
for tau in ("tau0.7", "tau0.8", "tau0.9"):
    det = round3["relative_rank"][tau]["detail"]  # 17 layers: k, co, ci
    rows = []
    tot = {kk: 0 for kk in ("A_p", "A_a", "B_p", "B_a", "ofA_p", "ofA_a", "ofB_p", "ofB_a")}
    worst_cp, worst_ca = 0.0, 0.0
    n_line_multiple = 0
    c_adv_layers = 0  # layers where aligned worst-case amp < packed
    sum_cp, sum_ca = 0.0, 0.0
    for d in det:
        r = all_models(d["k"], d["co"])
        rows.append({"k": d["k"], "co": d["co"], "ci": d["ci"], **r})
        tot["A_p"] += r["A_packed"]["lines"];  tot["ofA_p"] += r["A_packed"]["overfetch"]
        tot["A_a"] += r["A_aligned"]["lines"]; tot["ofA_a"] += r["A_aligned"]["overfetch"]
        tot["B_p"] += r["B_packed"]["lines"];  tot["ofB_p"] += r["B_packed"]["overfetch"]
        tot["B_a"] += r["B_aligned"]["lines"]; tot["ofB_a"] += r["B_aligned"]["overfetch"]
        worst_cp = max(worst_cp, r["C"]["packed_amp"])
        worst_ca = max(worst_ca, r["C"]["aligned_amp"])
        sum_cp += r["C"]["packed_amp"]; sum_ca += r["C"]["aligned_amp"]
        if r["C"]["aligned_amp"] < r["C"]["packed_amp"] - 1e-9:
            c_adv_layers += 1
        if r["B_mod_64"] == 0:
            n_line_multiple += 1
    AGG[tau] = {"n_layers": 17, "totals": tot, "worst_C_packed": worst_cp,
                "worst_C_aligned": worst_ca, "n_line_multiple_of_64B": n_line_multiple,
                "mean_C_packed": sum_cp / 17, "mean_C_aligned": sum_ca / 17,
                "n_layers_C_advantage": c_adv_layers,
                "detail": rows}
    t = tot
    print(f"{tau}: 17 ptw stages  A packed {t['A_p']} lines/{t['ofA_p']} B  vs aligned {t['A_a']}/{t['ofA_a']}"
          f"  |  B packed {t['B_p']}/{t['ofB_p']} vs aligned {t['B_a']}/{t['ofB_a']}"
          f"  | line-multiple layers {n_line_multiple}/17"
          f"  | mean C {sum_cp/17:.2f}x vs {sum_ca/17:.2f}x (adv {c_adv_layers}/17)"
          f"  | worst C {worst_cp:.2f}x vs {worst_ca:.2f}x")

# ---------- 3. MobileNetV2 stem pointwise (13->32, Co=32) ----------
STEM = {}
for k in (7, 13):
    r = all_models(k, 32)
    STEM[k] = r
    print(f"stem k={k} Co=32: A {r['A_packed']['lines']}/{r['A_packed']['overfetch']}"
          f" vs {r['A_aligned']['lines']}/{r['A_aligned']['overfetch']}"
          f"  B {r['B_packed']['lines']}/{r['B_packed']['overfetch']}"
          f" vs {r['B_aligned']['lines']}/{r['B_aligned']['overfetch']}"
          f"  C {r['C']['packed_amp']:.2f}x/{r['C']['aligned_amp']:.2f}x")

# ---------- 4. no-growth cap demo on the artificial narrow-output layer (S3) ----------
# Co=8, Ci=64, 3x3: baseline footprint 18432 B; raw energy criterion picks k=16
# (2.03x growth). Cap: k <= 9*Co*Ci/(9*Ci+Co)
Co, Ci = 8, 64
cap = 9 * Co * Ci / (9 * Ci + Co)
k_cap = math.floor(cap)
base_fp = 9 * Ci * 4 * Co  # single conv, spatial stage only (weights, no hw)
def two_stage_fp(k, Co, Ci):
    return 9 * Ci * 4 * k + k * Co * 4  # spatial (Ci->k,3x3) + pointwise (k->Co)
demo = {"cap": cap, "k_raw": 16, "fp_raw": two_stage_fp(16, Co, Ci),
        "k_capped": k_cap, "fp_capped": two_stage_fp(k_cap, Co, Ci),
        "baseline_fp": base_fp}
print("cap demo:", demo)

# ---------- 5. parametric timing example (S1) ----------
# effective DRAM path 3.2 GB/s; time = bytes / bw
BW = 3.2e9
r13 = all_models(13, N_REF)
timing = {
    "bw_GBps": 3.2,
    "k13_streaming_packed_us": (r13["B_packed"]["lines"] * L) / BW * 1e6,
    "k13_streaming_aligned_us": (r13["B_aligned"]["lines"] * L) / BW * 1e6,
    "k13_perchannel_packed_us": (r13["A_packed"]["lines"] * L) / BW * 1e6,
    "k13_perchannel_aligned_us": (r13["A_aligned"]["lines"] * L) / BW * 1e6,
}
print("timing:", timing)

out = {"validation": {str(k): v for k, v in VALIDATION.items()}, "validation_pass": ok,
       "aggregation": AGG, "stem": {str(k): v for k, v in STEM.items()},
       "cap_demo": demo, "timing": timing}
with open(r"./round4_fixes.json", "w") as f:
    json.dump(out, f, indent=1)
print("written round4_fixes.json")
