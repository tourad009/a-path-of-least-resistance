"""Complete, replayable record of one team game (nothing truncated).

A Transcript is filled by core.run_episode (and the post-game readers) and saved as gzipped JSON:
  header   cfg, git hash, roster, holders, difficulty, leak file, the keys (analyst only: agents never see this)
  agents   per agent: the full system brief and every file in its starting workspace
  events   in order:
           round_start  messages the game injected (share_first, teammate_asks)
           call         one model call: the exact observation sent, the full reply, the parsed command, tokens, cost, latency
           command      the command, its output as the agent saw it, every sandbox call it made (full shell output up to
                        core.FULL_OUTPUT_CHARS, code runs), the agent's score after it, messages it sent, leak opens and new
                        crossings (leak = v1, honest = v2)
           round_end    every score and item board, new crossings
  readers  post-game model calls: security officer (both views), privileged observer, interviews: full prompt and reply
  score    the game's final score dict; `post` holds the parsed officer / observer results added by the runner

replay(data) rebuilds the game from cfg and re-applies every command, with each sandbox call answered from the
transcript; it must give the recorded score, which shows the transcript is a complete record of the game.

    python -m harness.transcript <file.json.gz>     # replay check
"""
import base64
import datetime
import gzip
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

SCHEMA = 1


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        return None


def _jsonable(x):
    if isinstance(x, (set, frozenset, tuple)): return list(x)
    return str(x)


class Transcript:
    def __init__(self, cfg=None, git=None):
        self.data = dict(schema=SCHEMA, cfg=dict(cfg or {}), git=git or git_hash(),
                         created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                         header={}, agents=[], events=[], readers=[], score=None, post={})
        self._pending = {}  # agent -> sandbox calls made by the command being run
        self._lock = threading.Lock()  # post-game readers run in parallel threads

    # ---- recording (called by core.run_episode)
    def _ev(self, kind, **kw):
        self.data["events"].append(dict(seq=len(self.data["events"]), type=kind, **kw))

    def start(self, task, mode=None, agent_model=None):
        hp = task.hp() if hasattr(task, "hp") else False
        names = task.team_names
        self.data["header"] = dict(
            task=task.name, seed=task.seed, mode=mode, agent_model=agent_model, n_teams=task.n_teams, team_size=task.team_size,
            rounds=task.rounds, phases=list(task.phases()), difficulty=getattr(task, "difficulty", None),
            leak_file=task.leak_file, channel=task.channel,
            levers={k: getattr(task, k) for k in getattr(task, "levers", {})} | {"off": sorted(getattr(task, "levers", {}))},
            roster=[dict(agent=task.alias[i], team=names[task.team(i)], holder=task.is_holder(i)) for i in range(task.n_total)],
            holder_teams=sorted(names[t] for t in task.team_holder),
            # analyst only: the leak's values (v1) and, in honest-path games, the current key (v2)
            leak_values={names[t]: list(task.secret_values(t)) for t in task.team_holder} if hasattr(task, "secret_values") else {},
            honest_values={names[t]: list(task.honest_values(t)) for t in range(task.n_teams)} if hp else {},
        )
        self.data["agents"] = [dict(agent=task.alias[i], team=names[task.team(i)], holder=task.is_holder(i), brief=task.brief(i),
                                    workspace=task.workspace(i) if hasattr(task, "workspace") else task.sandbox_files(i))
                               for i in range(task.n_total)]

    def state(self, task):
        return dict(n_msgs=len(task.messages), cross=set(task.cross), hcross=set(getattr(task, "hcross", {})), opened=set(task.opened))

    def _msg(self, task, k, m):
        names, a = task.team_names, task.alias
        to = "all" if m["to"] is None else "team" if m["to"] == "team" else a[m["to"]]
        v1 = [names[t] for t in task.team_holder if task.leak_in_text(t, m["text"])]
        v2 = ([names[t] for t in range(task.n_teams) if task.honest_in_text(t, m["text"])]
              if getattr(task, "hp", lambda: False)() else [])
        return dict(id=k, round=m["r"], sender=a[m["frm"]], sender_team=names[task.team(m["frm"])], to=to,
                    team=names[m["team"]] if m["to"] == "team" else None, text=m["text"], carries_leak=v1, carries_honest=v2)

    def _diff(self, task, before):
        a, names = task.alias, task.team_names
        msgs = [self._msg(task, k, task.messages[k]) for k in range(before["n_msgs"], len(task.messages))]
        cross = [dict(agent=a[i], team=names[t], kind="leak", own_team=task.team(i) == t)
                 for (i, t) in sorted(set(task.cross) - before["cross"])]
        cross += [dict(agent=a[i], team=names[task.team(i)], kind="honest") for i in sorted(set(getattr(task, "hcross", {})) - before["hcross"])]
        opened = [a[i] for i in sorted(set(task.opened) - before["opened"])]
        return dict(messages=msgs, crossings=cross, opened_leak=opened)

    def call(self, r, i, obs, reply, cmd, meta, parse):
        meta = meta or {}
        self._ev("call", round=r, agent_index=i, observation=obs, reply=reply, command=cmd, parse=parse,
                 **{k: meta.get(k) for k in ("finish_reason", "prompt_tokens", "completion_tokens", "cost_usd", "latency_s",
                                             "refused", "error", "attempts")})

    def command(self, task, r, i, cmd, out, before):
        self._ev("command", round=r, agent_index=i, agent=task.alias[i], command=cmd, output=out,
                 sandbox=self._pending.pop(i, []), score_after=task.agent_score(i), **self._diff(task, before))

    def round_start(self, task, r, before):
        self._ev("round_start", round=r, **self._diff(task, before))

    def round_end(self, task, r, before):
        items = [task.item_status(i) for i in range(task.n_total)]
        self._ev("round_end", round=r, scores={task.alias[i]: task.agent_score(i) for i in range(task.n_total)},
                 items={task.alias[i]: items[i] for i in range(task.n_total)}, **self._diff(task, before))

    def sandbox_call(self, i, rec):
        self._pending.setdefault(i, []).append(rec)

    def reader(self, kind, model, prompt, reply, meta, agent=None):
        meta = meta or {}
        with self._lock:
            self.data["readers"].append(dict(kind=kind, model=model, agent=agent, prompt=prompt, reply=reply,
                                             **{k: meta.get(k) for k in ("finish_reason", "prompt_tokens", "completion_tokens",
                                                                         "cost_usd", "latency_s", "refused", "error")}))

    def finish(self, score):
        self.data["score"] = json.loads(json.dumps(score, default=_jsonable))

    def add_post(self, key, value):
        self.data["post"][key] = json.loads(json.dumps(value, default=_jsonable))

    # ---- storage
    def to_bytes(self):
        return gzip.compress(json.dumps(self.data, default=_jsonable).encode(), compresslevel=9)

    def to_b64(self):
        return base64.b64encode(self.to_bytes()).decode()


def load(path_or_bytes):
    raw = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else Path(path_or_bytes).read_bytes()
    return json.loads(gzip.decompress(raw))


def save_b64(b64, path):
    """Write a transcript sent back by the runner; returns its sha256 (stored in the results line)."""
    raw = base64.b64decode(b64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


# ---- sandboxes
class RecordingSandbox:
    """Wraps an agent's sandbox: every shell command and code run is recorded with its full output."""
    def __init__(self, inner, transcript, i):
        self.inner, self.t, self.i = inner, transcript, i

    def write(self, path, content): self.inner.write(path, content)

    def exec(self, cmd):
        if hasattr(self.inner, "last_full"): self.inner.last_full = None
        out = self.inner.exec(cmd)
        full = getattr(self.inner, "last_full", None)
        self.t.sandbox_call(self.i, dict(kind="exec", cmd=cmd, output=out, full_output=full if full and full != out else None))
        return out

    def run_python(self, script):
        out = self.inner.run_python(script)
        self.t.sandbox_call(self.i, dict(kind="python", script=script, output=out))
        return out

    def close(self): self.inner.close()


class ReplaySandbox:
    """Answers each sandbox call from the transcript, in order (checks the calls are the same)."""
    def __init__(self, calls):
        self.calls = list(calls)

    def write(self, path, content): pass

    def _next(self, kind, key, value):
        assert self.calls, f"replay made an extra {kind} call: {value[:80]}"
        c = self.calls.pop(0)
        assert c["kind"] == kind and c[key] == value, f"replay diverged: expected {c['kind']} {c.get(key, '')[:80]}"
        return c["output"]

    def exec(self, cmd): return self._next("exec", "cmd", cmd)
    def run_python(self, script): return self._next("python", "script", script)
    def close(self): pass


def replay(data):
    """Rebuild the game from its cfg and re-apply every recorded command. Returns (score, matches the recorded score)."""
    from harness.tasks import make_task
    task = make_task(data["cfg"])
    calls = [[] for _ in range(task.n_total)]
    for e in data["events"]:
        if e["type"] == "command":
            calls[e["agent_index"]] += e["sandbox"]
    sbs = [ReplaySandbox(c) for c in calls]
    task.setup(sbs)
    for e in data["events"]:
        if e["type"] == "round_start": task.start_round(e["round"])
        elif e["type"] == "command": task.run_command(e["agent_index"], e["command"], sbs[e["agent_index"]])
        elif e["type"] == "round_end": task.end_round(e["round"])
    assert all(not s.calls for s in sbs), "replay left sandbox calls unused"
    score = json.loads(json.dumps(task.full_score(), default=_jsonable))
    want = {k: v for k, v in (data["score"] or {}).items() if k != "reply_health"}
    return score, score == want


if __name__ == "__main__":
    for p in sys.argv[1:]:
        _, ok = replay(load(p))
        print(f"{'OK  ' if ok else 'DIFF'} {p}")
