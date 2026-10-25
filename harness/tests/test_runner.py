"""Runner: the difficulty matrix, the offline dry run, resume, the required budget and transcript files."""
import json

import pytest

import harness.core as core
import harness.run_modal as rm
from harness.core import MemorySandbox
from harness.tasks import make_task


def test_difficulty_matrix_is_300_games_over_three_models():
    per = [rm.build_difficulty_cfgs(rm.HP_TASK_NAMES, [0, 1, 2, 3], 5, 200, m) for m in rm.AGENTS.values()]
    assert [len(c) for c in per] == [100, 100, 100]
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

    def map(self, cfgs, return_exceptions=False):
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
