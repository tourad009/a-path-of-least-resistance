"""Run one episode locally with an LLM agent (no Modal, no sandbox):
  python -m harness.run_local <task> <seed> <on|off> [level] [pressure]
  e.g.  AGENT=meta-llama/llama-4-maverick python -m harness.run_local leaderboard 0 on 3"""
import json, os, sys
from harness.core import Cost, overseer, run_episode
from harness.tasks import BY_NAME

task_name, seed, chan = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "on"
level = int(sys.argv[4]) if len(sys.argv) > 4 else 0
pressure = len(sys.argv) > 5 and sys.argv[5] in ("1", "on", "true", "pressure")
AGENT, OVERSEER = os.environ.get("AGENT", "nousresearch/hermes-4-405b"), "openai/gpt-4o-mini"
cost = Cost()
task = BY_NAME[task_name](seed=seed, channel=chan, pressure=pressure, level=level)
ep, score, _ = run_episode(task, mode="llm", agent_model=AGENT, cost=cost)
ov = overseer(OVERSEER, task, ep, cost)
print(json.dumps({"task": task_name, "seed": seed, "channel": chan, "level": level, "pressure": pressure, "score": score,
                  "overseer": ov, "cost_usd": round(cost.usd, 4), "calls": cost.calls}, indent=2, default=str))
print("\n".join(ep.log[:40]))
