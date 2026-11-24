# -*- coding: utf-8 -*-
"""Round-2 figure refresh: fig1 line chart with 4-seed error bars, fig2 axis from 0.
Overwrites figures/ in paper-JSA. Uses the working-paper style of make_report_v2."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
PAP = r"./paper\figures"
os.makedirs(PAP, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

ms = json.load(open(os.path.join(OUT, "multiseed_results.json"), encoding="utf-8"))
kd = json.load(open(os.path.join(OUT, "kd_unified.json"), encoding="utf-8"))

# ---------------- fig1: accuracy across stages, 4 seeds ----------------
stages = ["fresh\ndecomposition", "after\nrecovery", "after\nKD polish", "after\nFCQ"]
taus = ["0.7", "0.8", "0.9"]
colors = {"0.7": "#c0392b", "0.8": "#e67e22", "0.9": "#2e86c1"}
teacher = 91.35

data = {t: {s: [] for s in range(4)} for t in taus}
# seed 42
for t in taus:
    s = kd["students"][f"svd_tau{t}"]
    data[t][0] = [s["acc_before_ft"], s["acc_ce"], s["acc_kd"], s["acc_kd_fcq"]]
# seeds 0,1,2
for i, k in enumerate(["0", "1", "2"], start=1):
    for t, key in [("0.7", "tau0.70"), ("0.8", "tau0.80"), ("0.9", "tau0.90")]:
        s = ms[k]["ours"][key]
        data[t][i] = [s["acc_before_ft"], s["acc_ce"], s["acc_kd"], s["acc_kd_fcq"]]

fig, ax = plt.subplots(figsize=(6.0, 3.6))
x = np.arange(4)
for t in taus:
    arr = np.array([data[t][i] for i in range(4)]) * 100   # seeds x stages
    mean, std = arr.mean(0), arr.std(0)
    ax.errorbar(x, mean, yerr=std, marker="o", capsize=4, lw=1.8,
                color=colors[t], label=f"$\\tau$={t} (4 seeds, mean$\\pm$std)")
ax.axhline(teacher, ls="--", lw=1.2, color="#555555",
           label=f"uncompressed teacher ({teacher:.2f}%)")
ax.set_xticks(x); ax.set_xticklabels(stages)
ax.set_ylabel("CIFAR-10 top-1 accuracy (%)")
ax.set_ylim(35, 94)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(PAP, "fig1_accuracy.png"), bbox_inches="tight")
plt.close(fig)
print("fig1 saved")

# ---------------- fig2: b8 latency, axis from 0 ----------------
res = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
base_lat = res["resnet18"]["baseline"]["latency_med_ms"] * res["resnet18"]["baseline"]["latency_ms"] / res["resnet18"]["baseline"]["latency_ms"]  # keep simple
labels = ["baseline"] + [f"$\\tau$={t}" for t in ["0.7", "0.8", "0.9"]]
lats = [res["resnet18"]["baseline"]["latency_ms"]]
for key in ["0.7", "0.8", "0.9"]:
    lats.append(res["resnet18"]["svd"][key]["latency_ms"])
lats = np.array(lats)
base = lats[0]
speed = base / lats
fig, ax = plt.subplots(figsize=(6.0, 3.4))
bars = ax.bar(labels, lats, color=["#7f8c8d", "#c0392b", "#e67e22", "#2e86c1"], width=0.55)
ax.set_ylabel("CPU latency per image (ms, batch 1, single thread)")
ax.set_ylim(0, lats.max() * 1.25)   # axis starts at 0 (round-2 m12)
for b, v, s in zip(bars, lats, speed):
    txt = f"{v:.1f}"
    if s > 1.01:
        txt += f"\n{s:.2f}x faster"
    elif s < 0.99:
        txt += f"\n{v/base:.2f}x latency vs baseline"
        b.set_edgecolor("#c0392b"); b.set_linewidth(2)
    ax.text(b.get_x() + b.get_width()/2, v + lats.max()*0.03, txt,
            ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(PAP, "fig2_latency.png"), bbox_inches="tight")
plt.close(fig)
print("fig2 saved")
