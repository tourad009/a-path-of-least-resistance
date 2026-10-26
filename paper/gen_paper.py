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
          "anthropic/claude-opus-5.5": (4.0, 20.0), "anthropic/claude-sonnet-5.5": (2.0, 10.0)}


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
             ModelHermes=rm.AGENTS["hermes"], ModelDolphin=rm.AGENTS["dolphin"], ModelOpus=rm.AGENTS["opus"],
             NTests=n_tests())
    for key, m in (("Hermes", rm.AGENTS["hermes"]), ("Dolphin", rm.AGENTS["dolphin"]), ("Opus", rm.AGENTS["opus"]),
                   ("Officer", rm.OVERSEERS["strong"])):
        d[f"PriceIn{key}"], d[f"PriceOut{key}"] = (f"{x:g}" for x in PRICES[m])
    for key, m in (("Hermes", "hermes"), ("Dolphin", "dolphin"), ("Opus", "opus")):
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
def results():
    p = ROOT / "analysis" / "numbers.json"
    models = [("Hermes-4-405B", "hermes-4-405b"), ("Dolphin-Mistral-24B", "dolphin-mistral-24b-venice-edition"),
              ("Claude Opus 5.5", "claude-opus-5.5")]
    res = json.loads(p.read_text()) if p.exists() else None
    ph = "\\torun{: H1}"
    fmt = lambda x: "--" if x is None else f"{x:.2f}"
    m = {}
    if res and "error" not in res["H1_teammate_use"]["gee"]:
        g, b = res["H1_teammate_use"]["gee"], res["H1_teammate_use"]["d3_minus_d0"]
        m = dict(HOneOR=f"{g['odds_ratio_per_level']:.2f}", HOneP=f"{g['p']:.3g}", HOneDiff=f"{100 * b['diff']:+.0f}",
                 HOneCI=f"{100 * b['ci95'][0]:+.0f} to {100 * b['ci95'][1]:+.0f}", HOneGames=str(res["n_games"]))
    for k in ("HOneOR", "HOneP", "HOneDiff", "HOneCI", "HOneGames"):
        m.setdefault(k, ph)
    rows = []
    for label, key in models:
        for d in (0, 1, 2, 3):
            cs = [c for k, c in (res or {}).get("cells", {}).items() if k.startswith(key + "|") and k.endswith(f"|{d}")]
            if cs:
                s = lambda a: sum(c[a] for c in cs)
                cells = [f"{int(s('used'))}/{int(s('teammates'))}", f"{int(s('sent'))}/{int(s('holders'))}",
                         f"{int(s('colluding_teams'))}/{int(s('holder_teams'))}",
                         fmt(s("control_honest_sum") / s("controls") if s("controls") else None)]
            else:
                cells = ["\\torun{}"] * 4
            rows.append(f"{label if d == 0 else ''} & D{d} & " + " & ".join(cells) + " \\\\\n")
        rows.append("\\midrule\n" if key != models[-1][1] else "")
    table = ("\\begin{tabular}{llcccc}\n\\toprule\nModel & Level & Teammates used & Holders sent & Collusion (teams) & "
             "Control honest rate \\\\\n\\midrule\n" + "".join(rows) + "\\bottomrule\n\\end{tabular}\n")
    return macros(m) + "\\newcommand{\\ResultsTable}{%\n" + table + "}\n"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in (("design", design), ("prompts", prompts), ("honest_paths", honest_paths), ("checks", checks),
                     ("smoke", smoke), ("results", results)):
        (OUT / f"{name}.tex").write_text(f"% generated by paper/gen_paper.py -- do not edit\n{fn()}")
        print(f"wrote {OUT / name}.tex")
