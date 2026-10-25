"""Play one honest-path game locally, without Modal (MemorySandbox: only ls, cat, head and pwd work in the shell).

    python -m harness.run_team_local <task> <seed> <difficulty> [honest|collude|llm] [-o game.json.gz]

honest / collude: scripted players (no model calls), to check a task and level end to end and look at its transcript.
llm: real agents through OpenRouter (AGENT=<model>, needs OPENROUTER_API_KEY); costs money. The game's transcript is
written with -o (default transcripts/local/<task>-s<seed>-d<level>-<mode>.json.gz); open it with harness.transcript_view.
"""
import argparse
import json
import os
from pathlib import Path

from harness.core import Cost, MemorySandbox, run_episode
from harness.tasks import make_task
from harness.transcript import Transcript, replay, load

ap = argparse.ArgumentParser()
ap.add_argument("task"); ap.add_argument("seed", type=int); ap.add_argument("difficulty", type=int)
ap.add_argument("mode", nargs="?", default="honest", choices=("honest", "collude", "llm"))
ap.add_argument("-o", "--out")
a = ap.parse_args()
cfg = dict(task=a.task, seed=a.seed, difficulty=a.difficulty, channel=True, v2=True, team_reward=True, uniform_holders=True,
           officer_prompt="neutral", agent=os.environ.get("AGENT", "nousresearch/hermes-4-405b") if a.mode == "llm" else None)
task, tr, cost = make_task(cfg), Transcript(cfg), Cost()
_, score, _ = run_episode(task, mode=a.mode, agent_model=cfg["agent"], cost=cost,
                          sandboxes=[MemorySandbox() for _ in range(task.n_total)], transcript=tr)
out = Path(a.out or f"transcripts/local/{a.task}-s{a.seed}-d{a.difficulty}-{a.mode}.json.gz")
out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(tr.to_bytes())
keys = ("difficulty", "n_holders", "holders_opened", "solo_ever", "holders_sent", "n_nonholders", "within_ever",
        "teams_colluded", "control_honest_rate", "agent_scores")
print(json.dumps({k: score.get(k) for k in keys}, indent=1))
print(f"replay matches: {replay(load(out))[1]}; cost ${cost.usd:.3f}; transcript -> {out}")
