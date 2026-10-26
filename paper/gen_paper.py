"""Generate the paper's numbers, prompts and tables from the harness itself, so the text cannot drift from the code.

    python paper/gen_paper.py        # writes paper/generated/*.tex (run by `make -C paper`)

design.tex        design numbers and settings (games, seeds, rounds, caps, models, officer, preset)
prompts.tex       verbatim prompt texts, taken from the harness constants and from a real brief
honest_paths.tex  for each task and level: README.txt and the public files that make up the honest path
checks.tex        scripted-player validation per task and level, chance bounds, the number of tests
smoke.tex         the smoke run, if its results file exists (a pipeline check, not a finding)
results.tex       the first finding's numbers from analysis/numbers.json, or placeholders until the runs exist
"""
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import core, teams  # noqa: E402
from harness import run_modal as rm  # noqa: E402
from harness.core import MemorySandbox, run_episode  # noqa: E402
from harness.tasks import make_task  # noqa: E402
from harness.team_tasks import HP_TASKS  # noqa: E402

OUT = Path(__file__).with_name("generated")
SEED = 200
FINAL = dict(channel=True, v2=True, team_reward=True, uniform_holders=True, officer_prompt="neutral")
SHORT = {"team_leaderboard_v2": "Leaderboard", "team_database": "Database", "team_auction": "Auction",
         "team_coding": "Coding", "team_doc_qa": "Document Q\\&A"}
# OpenRouter list prices, USD per million input / output tokens, read from openrouter.ai/api/v1/models on 2026-10-25
PRICES = {"nousresearch/hermes-4-405b": (1.0, 3.0), "cognitivecomputations/dolphin-mistral-24b-venice-edition": (0.2, 0.9),
          "anthropic/claude-opus-5.5": (4.0, 20.0), "anthropic/claude-sonnet-5.5": (2.0, 10.0),
          "deepseek/deepseek-v3.2": (0.28, 0.42), "meta-llama/llama-4-maverick": (0.188, 0.65),
          "anthropic/claude-haiku-5.5": (0.1, 0.5)}
# the six agent models: macro key, display name, short name used by the analysis, role (Table 3)
MODELS = [("Hermes", "Hermes-4-405B", "hermes", "large, low-safeguard post-training"),
          ("Maverick", "Llama-4-Maverick", "maverick", "open-weight instruct (Meta)"),
          ("DeepSeek", "DeepSeek-V3.2", "deepseek", "open-weight instruct, reasoning off"),
          ("Dolphin", "Dolphin-Mistral-24B", "dolphin", "small, low-safeguard (``uncensored'')"),
          ("Haiku", "Claude Haiku 5.5", "haiku", "frontier, high-safeguard, reasoning off"),
          ("Opus", "Claude Opus 5.5", "opus", "frontier, high-safeguard")]


def ascii(s):
    return (s.replace("\u2192", "->").replace("\u2014", "--").replace("\u2013", "-").replace("\u00d7", "x")
            .encode("ascii", "replace").decode())


def tex(s):
    """Escape text for LaTeX (outside verbatim)."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in ascii(s))


def verb(title, text):
    return f"\\begin{{promptbox}}{{{tex(title)}}}\n{ascii(text).rstrip()}\n\\end{{promptbox}}\n"


def macros(d):
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in d.items())


def task(name, d, seed=SEED):
    return make_task(dict(task=name, seed=seed, difficulty=d, **FINAL))


# ---------------------------------------------------------------- design
def design():
    per = {m: rm.build_difficulty_cfgs(rm.HP_TASK_NAMES, [0, 1, 2, 3], 5, 200, m) for m in rm.AGENTS.values()}
    n = sum(len(c) for c in per.values())
    t = task("team_coding", 0)
    lo, hi = teams.TeamTask.TEAM_RANGE
    d = dict(NGames=n, NGamesPerModel=len(next(iter(per.values()))), NModels=len(rm.AGENTS), NTasks=len(rm.HP_TASK_NAMES),
             NLevels=4, NSeeds=5, SeedRange="200--204", NRounds=t.rounds, NCmds=8, NFinalCmds=teams.LAST_CALL_CMDS,
             TeamSize=t.team_size, TeamsMin=lo, TeamsMax=hi, MsgCap=300, ShellCap=1500, MemoryLen=t.memory_len,
             MsgsShown=t.messages_shown, MaxTokens=400, Temperature="0.7", OfficerModel=rm.OVERSEERS["strong"],
             **{f"Model{k}": rm.AGENTS[short] for k, _, short, _ in MODELS},
             NTests=n_tests())
    for key, m in [(k, rm.AGENTS[short]) for k, _, short, _ in MODELS] + [("Officer", rm.OVERSEERS["strong"])]:
        d[f"PriceIn{key}"], d[f"PriceOut{key}"] = (f"{x:g}" for x in PRICES[m])
    for key, _, m, _ in MODELS:
        d[f"Est{key}"] = f"{rm.EST_PER_GAME[rm.AGENTS[m]] * 100:.0f}"
    d["EstTotal"] = f"{sum(rm.EST_PER_GAME.values()) * 100:.0f}"
    return macros({k: tex(str(v)) for k, v in d.items()})


def n_tests():
    out = subprocess.run([sys.executable, "-m", "pytest", "harness/tests", "--collect-only", "-q"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    return m[1] if m else "?"


# ---------------------------------------------------------------- prompts
def prompts():
    t2, t3 = task("team_leaderboard_v2", 2), task("team_leaderboard_v2", 3)
    h = t2.holder[0]
    brief = t2.brief(h)
    note = next(l for l in brief.splitlines() if l.startswith("Note: a file"))
    parts = [
        verb("System prompt (every agent, every call; {brief} is the agent's brief)", core.SYSTEM),
        verb("A holder's full brief (leaderboard task, level D2, seed 200)", brief),
        verb("Rules line (every brief)", teams.RULE_TEXT),
        verb("Goal line (every brief)", teams.TEAM_REWARD_GOAL),
        verb("Special items, levels D0-D2", t2.hp_rules("x")),
        verb("Special items, level D3", t3.hp_rules("the codes of your team's special items, kept in a confidential file")),
        verb("Holder note (holders only)", note),
        verb("Officer: honest-ceiling sentence, level D3", teams.CEILING_TEXT),
        verb("Officer: honest-ceiling sentence, levels D0-D2", teams.CEILING_TEXT_HP),
        verb("Officer: neutral instruction (replaces the original 'aggressive' one)", teams.NEUTRAL_LINE),
        verb("Officer prompt, full view (leaderboard, D2, roster and log omitted)",
             t2.security_prompt(core.Episode(t2)).split("ROSTER:")[0]),
        verb("Post-game interview (holders only)", core.DEBRIEF),
    ]
    return "".join(parts)


# ---------------------------------------------------------------- honest paths
def honest_paths():
    out = []
    for T in HP_TASKS:
        out.append(f"\\subsection*{{{SHORT[T.name]} (\\texttt{{{tex(T.name)}}})}}\n")
        t0 = task(T.name, 0)
        h = t0.holder[0]
        out.append(f"Leak file (holders only): \\texttt{{{tex(t0.leak_file)}}}, identical at every level.\n\n")
        for d in (0, 1, 2):
            t = task(T.name, d)
            base = set(t.sandbox_files(h))
            files = sorted(set(t.workspace(h)) - base - {"README.txt"}) or ["(no new file; see README)"]
            changed = sorted(f for f, c in t.hp_files(h).items() if f in base)
            names = ", ".join(f"\\texttt{{{tex(f)}}}" for f in files if not f.startswith("("))
            extra = (f"; changed: " + ", ".join(f"\\texttt{{{tex(f)}}}" for f in changed)) if changed else ""
            none = "public tables in the database" if T.name == "team_database" else "no new file"
            out.append(f"\\textbf{{D{d}}}: public honest-path files: {names or none}{extra}.\n")
            out.append(verb(f"README.txt at D{d}", t.hp_readme(h)))
        out.append("\\textbf{D3}: no README and no honest-path file; the brief says items 6--10 cannot be worked out from the public data.\n\n")
    return "".join(out)


# ---------------------------------------------------------------- checks
def checks():
    rows = []
    for T in HP_TASKS:
        cells = []
        for d in (0, 1, 2, 3):
            ctrl = n_ctrl = used = n_mates = honest_cross = 0
            for seed in range(200, 205):
                for mode in ("honest", "collude"):
                    t = task(T.name, d, seed)
                    _, sc, _ = run_episode(t, mode=mode, sandboxes=[MemorySandbox() for _ in range(t.n_total)])
                    if mode == "honest":
                        ctrl += sc["control_honest_rate"] * sc["n_control"]; n_ctrl += sc["n_control"]
                        honest_cross += sc["within_ever"] + sc["solo_ever"]
                    else:
                        used += sc["within_ever"]; n_mates += sc["n_nonholders"]
            cells.append(f"{ctrl / n_ctrl:.2f} / {honest_cross} / {used}/{n_mates}")
        rows.append(f"{SHORT[T.name]} & " + " & ".join(cells) + " \\\\\n")
    from harness.doc_qa import TeamDocQA
    body = ("\\begin{table}[h]\n\\centering\\small\n\\caption{\\textbf{Scripted players validate the scorer at every level} "
            "(seeds 200--204, no model calls). Each cell: honest players' solve rate on items 6--10 (control agents) / "
            "honest agents who crossed the leak threshold / colluding teammates who crossed it.}\n\\label{tab:checks}\n"
            "\\begin{tabular}{lcccc}\n\\toprule\nTask & D0 & D1 & D2 & D3 \\\\\n\\midrule\n" + "".join(rows)
            + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    mant, exp = f"{TeamDocQA.honest_crossing_bound():.0e}".split("e")
    return body + macros({"DocQABound": f"{mant}\\times10^{{{int(exp)}}}"})


# ---------------------------------------------------------------- smoke
def smoke():
    files = sorted(glob.glob(str(ROOT / "results" / "difficulty-*-smoke.jsonl")))
    if not files:
        return macros(dict(SmokeAvailable="0"))
    from harness.transcript import load, replay
    recs = [json.loads(l) for f in files for l in open(f) if l.strip()]
    ok = sum(replay(load(ROOT / r["transcript_path"]))[1] for r in recs)
    h = {k: sum(r["score"].get("reply_health", {}).get(k, 0) for r in recs) for k in ("calls", "no_block", "refused", "empty")}
    parsed = sum(bool(r.get("security", {}).get("parse_ok")) and bool(r.get("security_messages_only", {}).get("parse_ok")) for r in recs)
    model = recs[0]["cfg"]["agent"]
    return macros(dict(SmokeAvailable="1", SmokeModel=tex(model), SmokeGames=len(recs), SmokeCost=f"{sum(r['cost_usd'] for r in recs):.2f}",
                       SmokePerGame=f"{sum(r['cost_usd'] for r in recs) / len(recs):.3f}", SmokeReplayOK=ok,
                       SmokeCalls=h["calls"], SmokeNoBlock=h["no_block"], SmokeRefused=h["refused"], SmokeEmpty=h["empty"],
                       SmokeOfficerParsed=parsed))


# ---------------------------------------------------------------- results (the first finding)
def val(kind, key, sub, v):
    """A looked-up value: in the paper, \val{use}{Hermes}{3} expands it."""
    return f"\\expandafter\\def\\csname {kind}{key}{sub}\\endcsname{{{v}}}\n"


def results():
    """Every result number in the paper, from the main-run results files (the pre-registered exclusion applied)."""
    from harness.analysis_difficulty import analyse, broken, game_rows, gee_trend, load, paired_bootstrap, team_rows
    files = sorted(glob.glob(str(ROOT / "results" / "difficulty-*-main.jsonl")))
    if not files:
        return "\\newcommand{\\ResultsAvailable}{0}\n"
    raw = [json.loads(l) for f in files for l in open(f) if l.strip()]
    recs = load(files)
    excluded = dict(load.excluded)
    agents, holders = game_rows(recs)
    res = analyse(recs, n_boot=4000)
    pct = lambda x: "--" if x is None else f"{100 * x:.0f}"
    out = ["\\newcommand{\\ResultsAvailable}{1}\n"]
    m = dict(NGamesRecorded=len(raw), NGamesUsable=len(recs), NExcluded=sum(excluded.values()),
             TotalCost=f"{sum(r.get('cost_usd', 0) for r in raw):.0f}")
    # pooled tests (H1-H3) and false alarms by level (H4)
    for key, h, col in (("HOne", "H1_teammate_use", None), ("HTwo", "H2_holder_sending", None), ("HThree", "H3_holder_self_use", None)):
        g, b = res[h]["gee"], res[h]["d3_minus_d0"]
        m[f"{key}OR"] = f"{g['odds_ratio_per_level']:.2f}" if "p" in g else "--"
        m[f"{key}P"] = (f"{g['p']:.2g}" if g["p"] >= 1e-4 else "10^{" + str(int(__import__('math').floor(__import__('math').log10(g['p'])))) + "}") if "p" in g else "--"
        m[f"{key}Diff"] = f"{100 * b['diff']:+.0f}"
        m[f"{key}CI"] = f"{100 * b['ci95'][0]:+.0f} to {100 * b['ci95'][1]:+.0f}"
    m["HOneGames"] = str(res["n_games"])
    holm = res["holm_secondary"]
    for k, v in holm.items():
        m["Holm" + k.replace("_", "").replace("H2", "Two").replace("H3", "Three").replace("H4", "Four")] = "--" if v is None else f"{v:.2g}"
    for det, key in (("score_rule", "Rule"), ("officer_full", "Officer"), ("officer_messages", "Msgs")):
        g = res["H4_false_alarms"][det]
        m[f"FA{key}OR"] = f"{g['odds_ratio_per_level']:.2f}" if "p" in g else "--"
        m[f"FA{key}P"] = (f"{g['p']:.2g}" if g["p"] >= 1e-4 else "<10^{-4}") if "p" in g else "--"
    out.append(macros(m))
    # per model
    rows = []
    for key, name, short, _ in MODELS:
        mid = rm.AGENTS[short].split("/")[-1]
        a = [x for x in agents if x["model"] == mid]
        h = [x for x in holders if x["model"] == mid]
        mine = [r for r in recs if r["cfg"]["agent"] == rm.AGENTS[short]]
        n_ex = sum(v for k, v in excluded.items() if k.startswith(mid + "|"))
        out.append(val("games", key, "", len(mine)) + val("excl", key, "", n_ex))
        cells = []
        for d in range(4):
            ad, hd = [x for x in a if x["level"] == d], [x for x in h if x["level"] == d]
            ctl = [(r["score"]["control_honest_rate"], r["score"]["n_control"]) for r in mine
                   if r["score"]["difficulty"] == d and r["score"]["control_honest_rate"] is not None]
            use = sum(x["used"] for x in ad) / len(ad) if ad else None
            snt = sum(x["sent"] for x in hd) / len(hd) if hd else None
            slf = sum(x["self_use"] for x in hd) / len(hd) if hd else None
            hon = sum(c * n for c, n in ctl) / sum(n for _, n in ctl) if ctl else None
            out.append(val("use", key, d, pct(use)) + val("sent", key, d, pct(snt)) + val("self", key, d, pct(slf))
                       + val("honest", key, d, pct(hon)))
            cells.append((use, hon))
        g = gee_trend(a, "used") if a else {"error": "none"}
        b = paired_bootstrap(a, "used", n_boot=4000) if a else {"error": "none"}
        import math
        ok = "p" in g and math.isfinite(g["p"]) and g["odds_ratio_per_level"] < 1e3
        sep = not ok and any(x["used"] for x in a)  # use only at some levels: the trend model separates, no finite estimate
        trend = (f"$\\times${g['odds_ratio_per_level']:.2f} ($p={g['p']:.2g}$)" if ok
                 else "use only at D3" if sep and not any(x["used"] for x in a if x["level"] < 3) else "--")
        diff = f"{100 * b['diff']:+.0f} [{100 * b['ci95'][0]:+.0f}, {100 * b['ci95'][1]:+.0f}]" if "diff" in b else "--"
        out.append(val("trend", key, "", trend) + val("diff", key, "", diff))
        rows.append(f"{name} & {len(mine)} & " + " & ".join(pct(u) for u, _ in cells) + " & "
                    + " & ".join(pct(hh) for _, hh in cells) + f" & {trend} & {diff} \\\\\n")
    out.append("\\newcommand{\\ResultsTable}{%\n\\begin{tabular}{lrcccccccccc}\n\\toprule\n"
               "& & \\multicolumn{4}{c}{Teammates who used the leak (\\%)} & \\multicolumn{4}{c}{Control honest solve rate (\\%)} & & \\\\\n"
               "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
               "Model & Games & D0 & D1 & D2 & D3 & D0 & D1 & D2 & D3 & Trend per level & D3$-$D0 (pp) \\\\\n\\midrule\n"
               + "".join(rows) + "\\bottomrule\n\\end{tabular}}\n")
    # detection: false alarms on clean teams by level, and colluding teams caught, per detector (pooled)
    det = {k: {d: [0, 0] for d in range(4)} for k in ("score_rule", "officer_full", "officer_messages")}
    tp = {k: [0, 0] for k in det}
    for r in recs:
        d = r["score"]["difficulty"]
        for row in team_rows(r):
            k = row["detector"]
            if row["colluding"]: tp[k][0] += row["flagged"]; tp[k][1] += 1
            else: det[k][d][0] += row["flagged"]; det[k][d][1] += 1
    drows = []
    for k, label, key in (("score_rule", "Score rule (any member $>5$)", "Rule"), ("officer_full", "LLM officer, full view", "Officer"),
                          ("officer_messages", "LLM officer, messages only", "Msgs")):
        fpr = [det[k][d][0] / det[k][d][1] if det[k][d][1] else None for d in range(4)]
        for d in range(4): out.append(val("fpr", key, d, pct(fpr[d])))
        out.append(val("caught", key, "", f"{tp[k][0]}/{tp[k][1]}"))
        drows.append(f"{label} & {tp[k][0]}/{tp[k][1]} & " + " & ".join(pct(x) for x in fpr) + " \\\\\n")
    out.append("\\newcommand{\\DetectionTable}{%\n\\begin{tabular}{lccccc}\n\\toprule\n"
               "& Colluding teams & \\multicolumn{4}{c}{False alarms on clean teams (\\%)} \\\\\n\\cmidrule(lr){3-6}\n"
               "Detector & caught & D0 & D1 & D2 & D3 \\\\\n\\midrule\n" + "".join(drows) + "\\bottomrule\n\\end{tabular}}\n")
    # appendix: the full pipeline per model and level
    prow = []
    for key, name, short, _ in MODELS:
        mid = rm.AGENTS[short].split("/")[-1]
        for d in range(4):
            c = res["cells"]
            cs = [v for k, v in c.items() if k.startswith(mid + "|") and k.endswith(f"|{d}")]
            if not cs: continue
            S = lambda f: int(sum(x[f] for x in cs))
            prow.append(f"{name if d == 0 else ''} & D{d} & {S('games')} & {S('opened')}/{S('holders')} & {S('self_use')}/{S('holders')} & "
                        f"{S('sent')}/{S('holders')} & {S('used')}/{S('teammates')} & {S('colluding_teams')}/{S('holder_teams')} \\\\\n")
        prow.append("\\midrule\n")
    out.append("\\newcommand{\\PipelineTable}{%\n\\begin{tabular}{llcccccc}\n\\toprule\nModel & Level & Games & Holders opened & "
               "Holders used & Holders sent & Teammates used & Collusion (teams) \\\\\n\\midrule\n" + "".join(prow[:-1])
               + "\\bottomrule\n\\end{tabular}}\n")
    figures(agents, recs, det)
    return "".join(out)


def figures(agents, recs, det):
    """Figure 2: per model, teammate leak use and the control teams' honest solve rate by level (one 0-100% scale).
    Figure 3: false alarms on clean teams by level, per detector. Palette validated with the dataviz validator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e2dc"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.family": "serif"})
    fdir = Path(__file__).with_name("figures"); fdir.mkdir(exist_ok=True)
    SHORTNAME = dict(Hermes="Hermes-4-405B", Maverick="Llama-4-Maverick", DeepSeek="DeepSeek-V3.2", Dolphin="Dolphin-24B",
                     Haiku="Claude Haiku 5.5", Opus="Claude Opus 5.5")
    fig, axes = plt.subplots(1, 6, figsize=(7.2, 2.1), sharey=True)
    for ax, (key, name, short, _) in zip(axes, MODELS):
        mid = rm.AGENTS[short].split("/")[-1]
        a = [x for x in agents if x["model"] == mid]
        mine = [r for r in recs if r["cfg"]["agent"] == rm.AGENTS[short]]
        use, hon = [], []
        for d in range(4):
            ad = [x["used"] for x in a if x["level"] == d]
            ctl = [(r["score"]["control_honest_rate"], r["score"]["n_control"]) for r in mine
                   if r["score"]["difficulty"] == d and r["score"]["control_honest_rate"] is not None]
            use.append(100 * sum(ad) / len(ad) if ad else float("nan"))
            hon.append(100 * sum(c * n for c, n in ctl) / sum(n for _, n in ctl) if ctl else float("nan"))
        ax.plot(range(4), hon, color=BLUE, lw=2, marker="o", ms=4, label="honest work solves the special items")
        ax.plot(range(4), use, color=ORANGE, lw=2, marker="s", ms=4, label="teammates use the leak")
        ax.set_title(f"{SHORTNAME[key]}\n" + f"n = {len(mine)} games", fontsize=7.5, color=INK, linespacing=1.3)
        ax.set_xticks(range(4), ["D0", "D1", "D2", "D3"])
        ax.set_ylim(-4, 104); ax.set_yticks([0, 50, 100])
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of agents")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fdir / "fig_models.pdf", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for k, label, color, mk in (("score_rule", "score rule", ORANGE, "s"), ("officer_full", "officer, full view", BLUE, "o"),
                                ("officer_messages", "officer, messages only", AQUA, "^")):
        ys = [100 * det[k][d][0] / det[k][d][1] if det[k][d][1] else float("nan") for d in range(4)]
        ax.plot(range(4), ys, color=color, lw=2, marker=mk, ms=4.5, label=label)
    ax.set_xticks(range(4), ["D0\nwritten down", "D1", "D2", "D3\nno honest path"])
    ax.set_ylabel("clean teams flagged (%)"); ax.set_ylim(-3, 103)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_xlim(-0.2, 3.2)
    fig.tight_layout(); fig.savefig(fdir / "fig_detection.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in (("design", design), ("prompts", prompts), ("honest_paths", honest_paths), ("checks", checks),
                     ("smoke", smoke), ("results", results)):
        (OUT / f"{name}.tex").write_text(f"% generated by paper/gen_paper.py -- do not edit\n{fn()}")
        print(f"wrote {OUT / name}.tex")
