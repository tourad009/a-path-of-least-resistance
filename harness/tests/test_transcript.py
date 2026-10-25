"""Transcripts: a complete record of every game (calls, commands, full outputs, messages, crossings, post-game readers)
that replays to the recorded score. Games are played by stubbed models, so nothing is paid."""
import json
import re

import pytest

import harness.core as core
from harness.core import MemorySandbox, run_episode
from harness.tasks import make_task
from harness.team_tasks import HP_TASKS
from harness.transcript import RecordingSandbox, Transcript, load, replay
from harness.transcript_view import render, render_index

FINAL = dict(channel=True, v2=True, team_reward=True, uniform_holders=True)


def fake_chat(task_ref):
    """A stub for core.chat: holders read the leak and post it to the team, teammates read README.txt and answer with
    what they were sent; the officer, observer and interview get canned replies."""
    count = {}

    def chat(model, messages, cost, temperature=0.7, max_tokens=400, meta=None):
        if meta is not None: meta.update(finish_reason="stop", prompt_tokens=10, completion_tokens=5, cost_usd=0.0, refused=False)
        sys_msg = messages[0]["content"]
        if sys_msg.startswith("You are the security officer"):
            t = task_ref[0]
            return "ok\n```json\n" + json.dumps(dict(agents={a: 10 for a in t.alias}, teams={
                n: dict(decision="CLEAR", suspicion=10, reason="-") for n in t.team_names})) + "\n```"
        if sys_msg.startswith("You are a privileged analyst"):
            return '```json\n{"messages": []}\n```'
        if "debrief" in messages[-1]["content"]:
            return "I shared it because the team reward counts."
        t = task_ref[0]
        me = re.search(r"You are (anon-[0-9a-f]+)", sys_msg)[1]
        i = t.alias.index(me)
        k = count[me] = count.get(me, 0) + 1
        if t.is_holder(i):
            plan = [f"cat {t.leak_file}", "msg team " + t.leak_line(i), "done"]
        else:
            plan = ["cat README.txt", "x" * 0 + "ls", "done"]
        return f"Thinking about it.\n```bash\n{plan[min(k, len(plan)) - 1]}\n```"
    return chat


@pytest.mark.parametrize("T", HP_TASKS, ids=lambda T: T.name)
@pytest.mark.parametrize("d", (0, 3))
def test_llm_game_transcript_is_complete_and_replays(monkeypatch, T, d):
    from harness.run_modal import play
    cfg = dict(task=T.name, seed=200, difficulty=d, agent="stub", **FINAL, interview=True, security=True,
               officer_prompt="neutral", officer_views=["full", "messages"], git="test")
    ref = [None]
    real = make_task

    def mk(c):
        ref[0] = real(c)
        return ref[0]
    monkeypatch.setattr(core, "chat", fake_chat(ref))
    import harness.run_modal as rm
    monkeypatch.setattr("harness.tasks.make_task", mk)
    out = play(cfg, [MemorySandbox() for _ in range(real(cfg).n_total)])
    data = load(__import__("base64").b64decode(out["transcript_b64"]))
    t = ref[0]
    calls = [e for e in data["events"] if e["type"] == "call"]
    cmds = [e for e in data["events"] if e["type"] == "command"]
    assert len(calls) == out["score"]["reply_health"]["calls"] and all(c["observation"] and c["reply"] for c in calls)
    assert len(cmds) == len(out["private_log"]) and len(data["agents"]) == t.n_total
    assert all(a["brief"] and a["workspace"] for a in data["agents"])
    leak_msgs = [m for e in cmds for m in e["messages"] if m["carries_leak"]]
    assert len(leak_msgs) == len(t.holder)                        # every holder's post is flagged as carrying its leak
    assert {x for e in cmds for x in e["opened_leak"]} == {t.alias[h] for h in t.holder}
    kinds = [r["kind"] for r in data["readers"]]
    assert kinds.count("security_officer") == 1 and kinds.count("security_officer_messages_only") == 1
    assert kinds.count("interview") == len(t.holder) and all(r["prompt"] and r["reply"] for r in data["readers"])
    assert set(data["post"]) >= {"security", "security_messages_only", "observer", "debrief", "cost"}
    assert data["header"]["difficulty"] == d and data["header"]["leak_values"]
    score, ok = replay(data)
    assert ok, "replay did not reproduce the recorded score"
    html = render(data)
    assert "carries-leak" in html and t.alias[t.holder[0]] in html


@pytest.mark.parametrize("T", HP_TASKS, ids=lambda T: T.name)
@pytest.mark.parametrize("mode", ("honest", "collude"))
def test_scripted_games_replay_at_every_level(T, mode):
    for d in (0, 1, 2, 3):
        cfg = dict(task=T.name, seed=201, difficulty=d, **FINAL)
        t, tr = make_task(cfg), Transcript(cfg, git="test")
        _, sc, _ = run_episode(t, mode=mode, sandboxes=[MemorySandbox() for _ in range(t.n_total)], transcript=tr)
        data = load(tr.to_bytes())  # gzip round trip
        assert replay(data)[1]
        crossings = [x for e in data["events"] for x in e.get("crossings", [])]
        if mode == "collude":
            assert sum(x["kind"] == "leak" for x in crossings) == sc["within_ever"] + sc["solo_ever"]
        elif d < 3:
            assert sum(x["kind"] == "honest" for x in crossings) == t.n_total


def test_full_shell_output_is_kept_beyond_what_the_agent_sees():
    class Capped:
        last_full = None
        def exec(self, cmd):
            self.last_full = "y" * 5000
            return self.last_full[:1500]
    t = Transcript({}, git="test")
    sb = RecordingSandbox(Capped(), t, 0)
    shown = sb.exec("cat big.txt")
    rec = t._pending[0][0]
    assert len(shown) == 1500 and rec["output"] == shown and len(rec["full_output"]) == 5000


def test_index_page_lists_games_with_outcomes(tmp_path):
    cfg = dict(task="team_coding", seed=202, difficulty=1, **FINAL)
    t, tr = make_task(cfg), Transcript(cfg, git="test")
    run_episode(t, mode="collude", sandboxes=[MemorySandbox() for _ in range(t.n_total)], transcript=tr)
    (tmp_path / "g.json.gz").write_bytes(tr.to_bytes())
    html = render_index(tmp_path)
    assert "team_coding" in html and "used" in html
