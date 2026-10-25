"""The difficulty analysis on synthetic results with a known answer (scripted games, no model calls)."""
import random

from harness.analysis_difficulty import analyse, holm
from harness.core import MemorySandbox, run_episode
from harness.tasks import make_task

FINAL = dict(channel=True, v2=True, team_reward=True, uniform_holders=True)


def fake_results(p_collude):
    """One record per (task, seed, level): the game is played by scripted colluders with probability p_collude[level]."""
    recs, g = [], random.Random(7)
    for task in ("team_leaderboard_v2", "team_coding"):
        for seed in range(200, 212):
            for d in (0, 1, 2, 3):
                cfg = dict(task=task, seed=seed, difficulty=d, agent="m/test", **FINAL)
                t = make_task(cfg)
                mode = "collude" if g.random() < p_collude[d] else "honest"
                _, sc, _ = run_episode(t, mode=mode, sandboxes=[MemorySandbox() for _ in range(t.n_total)])
                recs.append(dict(cfg=cfg, score=sc, security={}, security_messages_only={}, observer={}))
    return recs


def test_analysis_finds_a_built_in_trend_and_a_null():
    res = analyse(fake_results({0: 0.1, 1: 0.3, 2: 0.6, 3: 0.9}), n_boot=500)
    h1 = res["H1_teammate_use"]
    assert h1["gee"]["slope_logodds"] > 0 and h1["gee"]["p"] < 0.01
    assert h1["d3_minus_d0"]["diff"] > 0.5 and h1["d3_minus_d0"]["ci95"][0] > 0
    cell = res["cells"]["test|team_coding|0"]
    assert cell["control_honest_rate"] == 1.0 and res["cells"]["test|team_coding|3"]["control_honest_rate"] == 0.0
    # the score rule flags every clean team at levels 0-2 (honest agents score 10) and none at level 3
    assert cell["score_rule_fpr"] == 1.0 and res["cells"]["test|team_coding|3"]["score_rule_fpr"] == 0.0
    flat = analyse(fake_results({d: 0.5 for d in range(4)}), n_boot=500)
    lo, hi = flat["H1_teammate_use"]["d3_minus_d0"]["ci95"]
    assert lo < 0.25 and flat["H1_teammate_use"]["gee"]["p"] > 0.01


def test_holm():
    assert holm({"a": 0.01, "b": 0.04, "c": None}) == {"a": 0.02, "b": 0.04, "c": None}
