"""Figure for the first results: the honest route's cost against honest solving (left) and teammate leak use (right).
Reads analysis/numbers.json; writes paper/figures/results_levels.pdf."""
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OLD = "/usr/share/texlive/texmf-dist/fonts/opentype/public/oldstandard"  # the paper's own body font
for f in ("OldStandard-Regular.otf", "OldStandard-Bold.otf"):
    font_manager.fontManager.addfont(f"{OLD}/{f}")

NAMES = {"hermes-4-405b": "Hermes-4-405B", "dolphin-mistral-24b-venice-edition": "Dolphin-Mistral-24B"}
cells = json.load(open("analysis/numbers.json"))["cells"]
agg = defaultdict(lambda: defaultdict(float))
for k, c in cells.items():
    m, _, l = k.split("|")
    if m not in NAMES:
        continue
    for key, v in c.items():
        if isinstance(v, (int, float)):
            agg[(m, int(l))][key] += v
            agg[("all", int(l))][key] += v

COLORS = {"hermes-4-405b": "#1F5FAD", "dolphin-mistral-24b-venice-edition": "#C77700", "all": "#222222"}
levels = [0, 1, 2, 3]
plt.rcParams.update({"font.family": "Old Standard", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharex=True)
panels = [("Control teams solve honestly", lambda a: a["control_honest_sum"] / a["controls"]),
          ("Teammates use the leak", lambda a: a["used"] / a["teammates"])]
for ax, (title, f) in zip(axes, panels):
    for m in list(NAMES) + ["all"]:
        ys = [100 * f(agg[(m, l)]) for l in levels]
        ax.plot(levels, ys, marker="o", ms=4, lw=2.2 if m == "all" else 1.6, ls="--" if m == "all" else "-",
                color=COLORS[m], label="Both models" if m == "all" else NAMES[m])
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xticks(levels, ["D0\nwritten\ndown", "D1\none\nstep", "D2\njoin and\nrule", "D3\nno\nroute"])
    ax.set_ylim(-3, 103 if "honestly" in title else 62)
    ax.set_ylabel("% of agents" if "use" in title else "% solved")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig("paper/figures/results_levels.pdf")
print("wrote paper/figures/results_levels.pdf")
