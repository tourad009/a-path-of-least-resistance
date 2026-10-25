"""Run the security officer and the privileged observer on games already played, without replaying the agents (only the officer's calls cost money).

    python -m harness.rereview <results.jsonl> [<out.jsonl>] [--messages-only] [--model <openrouter id>] [--officer-prompt neutral|aggressive]

--messages-only: the officer sees the messages but not the scores or submissions; the privileged observer's findings already
in the file are reused (they do not depend on what the officer sees), so only the officer is called.
--model: another officer model (e.g. a second vendor); --officer-prompt: override the prompt the games were played with.

Each game is rebuilt from its configuration and its messages are replayed from the log, so the officer sees exactly the
view it would have had live. Games recorded before the security-officer truth existed get it from their log."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from harness.core import Cost, Episode, NullSandbox, privileged_observer, security_review
from harness.tasks import make_task
from harness.teams import grade_security

MODEL = "anthropic/claude-sonnet-5.5"


def rebuild(rec, officer_prompt=None):
    """The task with its messages and score history restored, and the episode log."""
    task = make_task(dict(rec["cfg"], officer_prompt=officer_prompt) if officer_prompt else rec["cfg"])
    sb = NullSandbox()
    for r, i, cmd, _ in rec["private_log"]:
        if cmd.split(" ", 1)[0] == "msg":
            task.round = r
            task._run(i, cmd, sb)
    task.score_history = rec["score"]["score_history"]
    ep = Episode(task)
    ep.log = rec["log"]
    return task, ep


def review(rec, model=MODEL, messages_only=False, officer_prompt=None):
    task, ep = rebuild(rec, officer_prompt)
    score = dict(rec["score"])
    if "truth_rule_agents" not in score:  # older runs: anyone who used a leak, plus holders who passed theirs on
        senders = {task.alias[h] for h in task.holder for m in task.messages
                   if m["frm"] == h and m["to"] != h and task.leak_in_text(task.team(h), m["text"])}
        bad = set(score["truth_users"]) | senders
        team_of = {task.alias[i]: task.team_names[task.team(i)] for i in range(task.n_total)}
        score["truth_rule_agents"], score["truth_rule_teams"] = sorted(bad), sorted({team_of[a] for a in bad})
    cost = Cost()
    sec = security_review(model, task, ep, cost, messages_only)
    sec["grade"] = grade_security(sec, score)
    sec["messages_only"] = messages_only
    sec["model"], sec["officer_prompt"] = model, task.officer_prompt
    obs = dict(rec["observer"]) if messages_only and rec.get("observer", {}).get("parse_ok") else privileged_observer(model, task, cost)
    obs["summary"] = task.observer_summary(obs, score, sec)
    sec["cost_usd"] = cost.usd
    return dict(rec, security=sec, observer=obs, score=score)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--messages-only", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--officer-prompt", choices=("neutral", "aggressive"))
    a = ap.parse_args()
    mo, src = a.messages_only, a.src
    suffix = ("-security-msgs" if mo else "-security") + ("" if a.model == MODEL else "-" + a.model.split("/")[-1])
    out = a.out or src.replace(".jsonl", suffix + ".jsonl")
    recs = [json.loads(l) for l in open(src)]
    with ThreadPoolExecutor(8) as pool:
        done = list(pool.map(lambda r: review(r, model=a.model, messages_only=mo, officer_prompt=a.officer_prompt), recs))
    with open(out, "w") as f:
        for d in done: f.write(json.dumps(d) + "\n")
    print(f"{len(done)} games reviewed, ${sum(d['security']['cost_usd'] for d in done):.2f} -> {out}")
