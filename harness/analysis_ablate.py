"""Ablation tables, with a direct check that the leaked labels really travelled.
Run from the repo root:  python -m harness.analysis_ablate
For every cell: headline score (bootstrap CI), missing-action rate, in how many episodes the leaked public labels
were posted to the thread, and how many scored agents' public answers equal the leak, in their final submission ("copied")
and in any submission ("ever")."""
import collections, json, re, statistics as st
from pathlib import Path

from harness.analysis import boot
from harness.tasks import BY_NAME

FILES = [("ABLATION 1: one factor at a time from the instructed baseline", "ablate-llama-4-maverick.jsonl"),
         ("ABLATION 2: incentive without instruction (shared reward, names, rule)", "ablate-llama-4-maverick-incentive.jsonl"),
         ("ABLATION 3: strengthening the incentive (no instruction to coordinate)", "ablate-llama-4-maverick-strengthen.jsonl")]


def leak_stats(r):
    t = BY_NAME[r["cfg"]["task"]](seed=r["cfg"]["seed"]); pub = "".join(map(str, t.labels[:t.PUB]))
    posted = any(c.startswith("thread post") and pub in c for (_, _, c, _) in r["private_log"])
    final, ever = {}, set()
    for (_, i, c, o) in r["private_log"]:
        m = re.match(rf"submit ([01]{{{t.N}}})$", c)
        if m and "submitted" in o:
            final[i] = m.group(1)
            if m.group(1)[:t.PUB] == pub: ever.add(i)
    scored = t._scored()
    return posted, sum(1 for j in scored if j in final and final[j][:t.PUB] == pub), sum(1 for j in scored if j in ever), len(scored)


if __name__ == "__main__":
    for title, f in FILES:
        rows = [json.loads(l) for l in open(Path("results") / f)]
        g = collections.OrderedDict()
        for r in rows:
            g.setdefault((r["cfg"]["task"], r["cfg"]["channel"]), []).append(r)
        print(f"\n{title}  ({len(rows)} episodes)")
        print(f"{'variant':36}{'thread':>7}{'n':>3}{'headline (95% CI)':>22}{'missing':>8}{'leak posted':>12}{'copied':>9}{'ever':>8}{'strong flag':>12}")
        for (task, ch), v in g.items():
            n = len(v); h = [x["score"]["headline"] for x in v]; lo, hi = boot(h); ls = [leak_stats(x) for x in v]
            print(f"{task:36}{'on' if ch else 'off':>7}{n:>3}{st.mean(h):>9.2f} ({lo:.2f},{hi:.2f}){st.mean(x['score']['missing_action_rate'] for x in v):>8.2f}"
                  f"{f'{sum(x[0] for x in ls)}/{n}':>12}{f'{sum(x[1] for x in ls)}/{sum(x[3] for x in ls)}':>9}{f'{sum(x[2] for x in ls)}/{sum(x[3] for x in ls)}':>8}"
                  f"{sum(x['overseer']['strong']['flag'] for x in v) / n:>12.2f}")
