"""Runner: the difficulty matrix, the offline dry run, resume, the required budget and transcript files."""
import json

import pytest

import harness.core as core
import harness.run_modal as rm
from harness.core import MemorySandbox
from harness.tasks import make_task


def test_difficulty_matrix_is_300_games_over_three_models():
    per = [rm.build_difficulty_cfgs(rm.HP_TASK_NAMES, [0, 1, 2, 3], 5, 200, m) for m in rm.AGENTS.values()]
    assert all(len(c) == 100 for c in per) and len(per) == 6
    c = per[0]
    assert {x["seed"] for x in c} == {200, 201, 202, 203, 204} and {x["difficulty"] for x in c} == {0, 1, 2, 3}
    assert all(x["officer_prompt"] == "neutral" and x["team_reward"] and x["v2"] and x["channel"] and not x.get("surveillance")
               for x in c)
    # every cfg builds, at its level
    for x in c[:20]:
        assert make_task(x).difficulty == x["difficulty"]


def test_game_key_ignores_the_git_hash_and_id():
    a = dict(task="t", seed=1, difficulty=0, git="aaa", game_id="x")
    assert rm.game_key(a) == rm.game_key(dict(a, git="bbb", game_id="y")) != rm.game_key(dict(a, seed=2))
    assert rm.game_id(dict(task="team_coding", seed=201, difficulty=2), 0) == "team_coding-s201-d2"


class NoRemote:
    def map(self, *a, **k): raise AssertionError("a dry run must not call Modal")


def test_dry_run_makes_no_remote_call(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rm, "episode", NoRemote())
    rm.main.info.raw_f(mode="difficulty", agent="hermes", dry_run=True)
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "100 games" in out and "$25.00" in out
    assert not (tmp_path / "results" / "difficulty-hermes-4-405b.jsonl").exists()


def test_paid_launch_needs_a_budget(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rm, "episode", NoRemote())
    with pytest.raises(AssertionError, match="--budget"):
        rm.main.info.raw_f(mode="difficulty", agent="hermes", tasks="team_coding", levels="0", seeds=1)


class LocalMap:
    """Plays games locally with scripted (stubbed) agents instead of on Modal."""
    def __init__(self): self.seen = []

    def map(self, cfgs, return_exceptions=False, order_outputs=True):
        for c in cfgs:
            self.seen.append(c["game_id"])
            yield rm.play(c, [MemorySandbox() for _ in range(make_task(c).n_total)])


def test_resume_skips_finished_games_and_writes_transcripts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core, "chat", lambda *a, **k: "```bash\ndone\n```")  # every agent ends its turn at once
    local = LocalMap()
    monkeypatch.setattr(rm, "episode", local)
    kw = dict(mode="difficulty", agent="hermes", tasks="team_coding", levels="0,3", seeds=1, budget=1.0, tag="t")
    rm.main.info.raw_f(**kw, pairs="team_coding:200")
    out = tmp_path / "results" / "difficulty-hermes-4-405b-t.jsonl"
    first = [json.loads(l) for l in out.open()]
    assert len(first) == 2 and local.seen == ["team_coding-s200-d0", "team_coding-s200-d3"]
    rm.main.info.raw_f(**kw)  # same file exists, no --resume: refuse
    assert len(out.read_text().splitlines()) == 2
    local.seen.clear()
    rm.main.info.raw_f(**dict(kw, seeds=2), resume=True)
    assert local.seen == ["team_coding-s201-d0", "team_coding-s201-d3"]
    recs = [json.loads(l) for l in out.open()]
    assert len(recs) == 4
    from harness.transcript import load, replay
    for r in recs:
        p = tmp_path / r["transcript_path"]
        assert p.exists() and "transcript_b64" not in r
        assert replay(load(p))[1]


def test_one_launch_plays_several_models_into_one_file_each(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core, "chat", lambda *a, **k: "```bash\ndone\n```")
    local = LocalMap()
    monkeypatch.setattr(rm, "episode", local)
    rm.main.info.raw_f(mode="difficulty", agent="hermes,opus", tasks="team_coding", levels="0", seeds=2, budget=1.0, tag="m")
    for name in ("hermes-4-405b", "claude-opus-5.5"):
        recs = [json.loads(l) for l in (tmp_path / "results" / f"difficulty-{name}-m.jsonl").open()]
        assert len(recs) == 2 and all(r["cfg"]["agent"].endswith(name) for r in recs)
    assert len(local.seen) == 4


def test_longest_games_start_first():
    cfgs = rm.build_difficulty_cfgs(["team_coding"], [0], 3, 200, rm.AGENTS["dolphin"]) + \
           rm.build_difficulty_cfgs(["team_coding"], [0], 3, 200, rm.AGENTS["opus"])
    order = rm.longest_first(cfgs)
    assert [c["agent"] for c in order[:3]] == [rm.AGENTS["opus"]] * 3
    sizes = [make_task(c).n_total for c in order[:3]]
    assert sizes == sorted(sizes, reverse=True)


class CostlyMap(LocalMap):
    def map(self, cfgs, return_exceptions=False, order_outputs=True):
        for r in super().map(cfgs, return_exceptions, order_outputs):
            r["cost_usd"] = 0.6
            yield r


def test_budget_stops_the_run(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core, "chat", lambda *a, **k: "```bash\ndone\n```")
    local = CostlyMap()
    monkeypatch.setattr(rm, "episode", local)
    rm.main.info.raw_f(mode="difficulty", agent="dolphin", tasks="team_coding", levels="0,1,2,3", seeds=1, budget=1.0, tag="b")
    assert "STOP: budget" in capsys.readouterr().out and len(local.seen) == 2


def test_failed_games_are_reported_and_left_for_resume(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    class Failing:
        def map(self, cfgs, return_exceptions=False, order_outputs=True):
            for _ in cfgs: yield RuntimeError("sandbox refused")
    monkeypatch.setattr(rm, "episode", Failing())
    rm.main.info.raw_f(mode="difficulty", agent="dolphin", tasks="team_coding", levels="0", seeds=2, budget=1.0, tag="f")
    out = capsys.readouterr().out
    assert "2 failed" in out and "--resume" in out


def test_rate_limits_are_retried_then_fail_loudly(monkeypatch):
    """A 429 is retried; one that never clears raises APIFailure instead of returning an empty reply (read as done)."""
    class R:
        def __init__(self, code, body): self.status_code, self._b, self.headers = code, body, {}
        ok = property(lambda self: self.status_code == 200)
        text = property(lambda self: json.dumps(self._b))
        def json(self): return self._b
    good = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"cost": 0.01, "total_tokens": 5}}
    calls = iter([R(429, {}), R(503, {}), R(200, good)])
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: next(calls))
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    out = core.chat_full("m", [], core.Cost())
    assert out["content"] == "hi" and out["retries"] == 2
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: R(429, {}))
    with pytest.raises(core.APIFailure):
        core.chat_full("m", [], core.Cost(), retry_budget_s=0)


def test_in_flight_credit_reservations_wait_but_real_credit_errors_stop(monkeypatch):
    class R:
        def __init__(self, code, body): self.status_code, self._b, self.headers = code, body, {}
        ok = property(lambda self: self.status_code == 200)
        text = property(lambda self: json.dumps(self._b))
        def json(self): return self._b
    good = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}], "usage": {"cost": 0.0}}
    reserve = R(402, {"error": {"message": "would exceed your available credits given your current in-flight requests"}})
    calls = iter([reserve, reserve, R(200, good)])
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: next(calls))
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    out = core.chat_full("m", [], core.Cost())
    assert out["retries"] == 2 and out["refused"]          # the content filter counts as a refusal
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: R(402, {"error": {"message": "Insufficient credits"}}))
    with pytest.raises(RuntimeError, match="402"):
        core.chat_full("m", [], core.Cost())


def test_reasoning_models_get_reasoning_switched_off(monkeypatch):
    sent = {}
    class R:
        status_code, headers, ok, text = 200, {}, True, ""
        def json(self): return {"choices": [{"message": {"content": "```bash\ndone\n```"}, "finish_reason": "stop"}], "usage": {}}
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(core.requests, "post", lambda url, json=None, **k: sent.update(json) or R())
    core.LLMAgent("deepseek/deepseek-v3.2", core.Cost()).act("b", "o")
    assert sent["reasoning"] == {"enabled": False} and sent["max_tokens"] == 400 and sent["provider"] == {"sort": "latency"}
    sent.clear()
    core.LLMAgent("nousresearch/hermes-4-405b", core.Cost()).act("b", "o")
    assert "reasoning" not in sent
