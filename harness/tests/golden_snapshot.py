"""Legacy golden snapshot: what the imported harness (collusion-leaderboard@73c4aa1) produces for the four harmonised team
tasks, before any honest-path change. test_difficulty.py checks that games built without `difficulty` still match it exactly.

    python -m harness.tests.golden_snapshot   # rewrite the snapshot (only when a legacy change is intended)
"""
import json
from pathlib import Path

from harness.core import Episode, MemorySandbox, run_episode
from harness.tasks import make_task

PATH = Path(__file__).with_name("golden_legacy.json")
TASKS = ("team_leaderboard_v2", "team_coding", "team_database", "team_auction")
SEEDS = (0, 1, 2, 3)
PRESETS = {
    "default": {},
    "final": dict(v2=True, team_reward=True, uniform_holders=True),
}


def _cfg(task, seed, preset):
    return dict(task=task, seed=seed, channel=True, **PRESETS[preset])


def snapshot():
    out = {}
    for name in TASKS:
        for seed in SEEDS:
            for preset in PRESETS:
                task = make_task(_cfg(name, seed, preset))
                rec = dict(briefs=[task.brief(i) for i in range(task.n_total)],
                           files=[task.sandbox_files(i) for i in range(task.n_total)],
                           observation0=task.observation(0, []),
                           security_prompt=task.security_prompt(Episode(task)),
                           security_msg_prompt=task.security_prompt(Episode(task), messages_only=True))
                for mode in ("honest", "collude"):
                    t = make_task(_cfg(name, seed, preset))
                    _, score, _ = run_episode(t, mode=mode, sandboxes=[MemorySandbox() for _ in range(t.n_total)])
                    rec[f"score_{mode}"] = json.loads(json.dumps(score))
                out[f"{name}|{seed}|{preset}"] = rec
    return out


if __name__ == "__main__":
    PATH.write_text(json.dumps(snapshot(), indent=1, sort_keys=True))
    print(f"wrote {PATH} ({PATH.stat().st_size} bytes)")
