# -*- coding: utf-8 -*-
"""Round-4 m8: regenerate fig1 with a broken y-axis (0 -> 94 with a break
between 52 and 60), so the truncation is explicit. Same data as round 2."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"."
PAP = r"./paper\figures"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

ms = json.load(open(os.path.join(OUT, "multiseed_results.json"), encoding="utf-8"))
kd = json.load(open(os.path.join(OUT, "kd_unified.json"), encoding="utf-8"))

stages = ["fresh\ndecomposition", "after\nrecovery", "after\nKD polish", "after\nFCQ"]
taus = ["0.7", "0.8", "0.9"]
colors = {"0.7": "#c0392b", "0.8": "#e67e22", "0.9": "#2e86c1"}
teacher = 91.35

data = {t: {s: [] for s in range(4)} for t in taus}
for t in taus:
    s = kd["students"][f"svd_tau{t}"]
    data[t][0] = [s["acc_before_ft"], s["acc_ce"], s["acc_kd"], s["acc_kd_fcq"]]
for i, k in enumerate(["0", "1", "2"], start=1):
    for t, key in [("0.7", "tau0.70"), ("0.8", "tau0.80"), ("0.9", "tau0.90")]:
        s = ms[k]["ours"][key]
        data[t][i] = [s["acc_before_ft"], s["acc_ce"], s["acc_kd"], s["acc_kd_fcq"]]

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6.0, 4.4),
    gridspec_kw={"height_ratios": [3, 2], "hspace": 0.07})

x = np.arange(4)
for ax in (ax_top, ax_bot):
    for t in taus:
        arr = np.array([data[t][i] for i in range(4)]) * 100
        mean, std = arr.mean(0), arr.std(0)
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=4, lw=1.8,
                    color=colors[t], label=f"$\\tau$={t} (4 seeds, mean$\\pm$std)")
    ax.axhline(teacher, ls="--", lw=1.2, color="#555555",
               label=f"uncompressed teacher ({teacher:.2f}%)")

ax_top.set_ylim(50, 94)
ax_bot.set_ylim(0, 46)
ax_bot.set_xticks(x); ax_bot.set_xticklabels(stages)
ax_top.legend(frameon=False, fontsize=9, loc="lower right")
fig.supylabel("CIFAR-10 top-1 accuracy (%)", x=0.02)

# axis-break marks
d = 0.5
kw = dict(marker=[(-1, -d), (1, 0)], markersize=11, linestyle="none",
          color="k", mec="k", mew=1.2, clip_on=False)
ax_top.plot([0], [0], transform=ax_top.transAxes, **kw)
ax_top.plot([1], [0], transform=ax_top.transAxes, **kw)
ax_bot.plot([0], [1], transform=ax_bot.transAxes, **kw)
ax_bot.plot([1], [1], transform=ax_bot.transAxes, **kw)
ax_top.spines["bottom"].set_visible(False)
ax_top.tick_params(bottom=False)
ax_bot.spines["top"].set_visible(False)

fig.subplots_adjust(hspace=0.07)
fig.savefig(os.path.join(PAP, "fig1_accuracy.png"), bbox_inches="tight")
plt.close(fig)
print("fig1 (broken axis) saved")
