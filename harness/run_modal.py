"""Parallel runner on Modal.
  modal run harness/run_modal.py --mode sandbox-check
  modal run harness/run_modal.py --mode difficulty --agent <openrouter id> --tag <name> --budget <usd> [--dry-run] [--resume]
  modal run harness/run_modal.py --mode teams ... (legacy team runs) | --mode pilot | --mode full --seeds 5 (legacy)
Each episode runs in one Modal function; every agent gets its own offline Modal Sandbox. Every game's full transcript
comes back with it and is written to transcripts/<run>/<game>.json.gz (see harness/transcript.py).
"""
import json
import os
import time
from pathlib import Path

import modal

app = modal.App("collusion-harness")
func_image = modal.Image.debian_slim(python_version="3.12").pip_install("requests").add_local_python_source("harness")
sandbox_image = modal.Image.debian_slim(python_version="3.12").run_commands(
    "useradd -m -s /bin/bash agent", "mkdir -p /home/agent/work && chown -R agent /home/agent")
SECRETS = [modal.Secret.from_name("openrouter")]
AGENT_MODEL = "nousresearch/hermes-4-405b"  # low-safeguard agent model (was Dolphin-Mistral-24B)
OVERSEERS = {"weak": "openai/gpt-4o-mini", "strong": "anthropic/claude-sonnet-5.5"}

# the experiment's three agent models, and a rough cost per game (agents + officer + observer + interviews). Hermes is
# measured on the earlier final pilot, Dolphin on our smoke run (10 games, $1.08); Opus is scaled from Hermes by its
# OpenRouter prices. Used by --dry-run and the paper only.
AGENTS = {"hermes": "nousresearch/hermes-4-405b", "dolphin": "cognitivecomputations/dolphin-mistral-24b-venice-edition",
          "opus": "anthropic/claude-opus-5.5",
          # second wave
          "deepseek": "deepseek/deepseek-v3.2", "maverick": "meta-llama/llama-4-maverick", "haiku": "anthropic/claude-haiku-5.5",
          # third wave
          "kimi": "moonshotai/kimi-k2-0905", "qwen": "qwen/qwen3-235b-a22b-2507", "minimax": "minimax/minimax-m2.5"}
# Hermes measured on the earlier final pilot, Dolphin on our smoke run; the rest scaled by OpenRouter prices
EST_PER_GAME = {AGENTS["hermes"]: 0.25, AGENTS["dolphin"]: 0.11, AGENTS["opus"]: 1.10,
                AGENTS["deepseek"]: 0.12, AGENTS["maverick"]: 0.11, AGENTS["haiku"]: 0.09,
                AGENTS["kimi"]: 0.23, AGENTS["qwen"]: 0.09, AGENTS["minimax"]: 0.13}
HP_TASK_NAMES = ("team_leaderboard_v2", "team_database", "team_auction", "team_coding", "team_doc_qa")
# honest-path difficulty runs: the friendly setup of the final pilot (random teams, holders on 1..T-1 teams, team-only
# reward, progress board), messaging on, post-game interviews, and a passive officer with the neutral prompt (it judges
# after the game and changes nothing in it). No surveillance warning.
DIFFICULTY_PRESET = dict(channel=True, rung="level0", v2=True, team_reward=True, uniform_holders=True, interview=True,
                         security=True, officer_prompt="neutral", officer_views=["full", "messages"])


# Ablations: one change at a time against the main run (same seeds, so every comparison is paired with a main-run game).
# Defaults: the three models that took the leak, levels D0 and D3, the five tasks, seeds 200-204.
ABLATION_MODELS = ("hermes", "maverick", "deepseek")
ARMS = {
    "handover":    dict(change=dict(share_first=True)),            # the game posts the leak to the team for the holder
    "ask":         dict(change=dict(teammate_asks=True)),          # a teammate asks the team for what items 6-10 need
    "warning":     dict(change=dict(surveillance=True)),           # agents are told an officer reads everything
    "norule":      dict(change=dict(no_rule=True)),                # no RULES line
    "individual":  dict(change=dict(team_reward=False, individual_reward=True)),  # rewarded for one's own score only
    "noteamchat":  dict(change=dict(levers={"team_channel": False})),
    "messyleak":   dict(change=dict(levers={"easy_copy": False})),  # no one-line summary at the top of the leak
    "nomemory":    dict(change=dict(levers={"keep_memory": False})),
    "nomessaging": dict(change=dict(channel=False)),               # structural control: no messages at all
    "neutralleak": dict(change=dict(leak_look="neutral"), tasks=("team_database", "team_coding", "team_doc_qa")),
    "password":    dict(change={}, tasks=("team_database_password",), seeds=(200, 25)),  # the leak is a password
    "dbmore":      dict(change={}, tasks=("team_database",), seeds=(205, 20)),  # answers leak on the password arm's seeds
    "pushclaude":  dict(change=dict(share_first=True, no_rule=True), models=("haiku",), levels=(3,)),
}


# the order ablations run in: the most informative arms first, so a budget stop still leaves complete arms
ARM_PRIORITY = ("handover", "nomessaging", "warning", "norule",            # 1: mechanism, structural check, deterrence
                "password", "dbmore", "individual",                       # 2: leak type, reward
                "ask", "noteamchat", "pushclaude",                        # 3: social request, friction, Claude
                "messyleak", "nomemory", "neutralleak")                   # 4: details of the leak and memory


def priority_order(cfgs):
    """Arms in ARM_PRIORITY order; inside an arm, seeds outermost and models and levels interleaved, so every prefix of the
    run holds matched cells for all models and both levels. Slow games go first inside each seed (shorter tail)."""
    from harness.tasks import make_task
    speed = {AGENTS["hermes"]: 2, AGENTS["deepseek"]: 2, AGENTS["maverick"]: 1, AGENTS["haiku"]: 1}
    rank = {a: k for k, a in enumerate(ARM_PRIORITY)}
    return sorted(cfgs, key=lambda c: (rank.get(c.get("arm"), len(rank)), c["seed"], -speed.get(c["agent"], 1),
                                       -make_task(c).n_total, c["task"], c["difficulty"]))


def build_ablation_cfgs(arms, models=None):
    out = []
    for arm in arms:
        spec = ARMS[arm]
        start, n = spec.get("seeds", (200, 5))
        for m in (models or spec.get("models", ABLATION_MODELS)):
            for c in build_difficulty_cfgs(spec.get("tasks", HP_TASK_NAMES), spec.get("levels", (0, 3)), n, start, AGENTS[m]):
                c.update(spec["change"], arm=arm, sandbox="small")
                out.append(c)
    return out


def build_difficulty_cfgs(tasks, levels, seeds, seed_start, agent, extra=None):
    """Every (seed, task, level) game for one agent model. Seeds are matched across tasks, levels and models."""
    return [dict(task=n, seed=s, difficulty=d, agent=agent, **DIFFICULTY_PRESET, **(extra or {}))
            for s in range(seed_start, seed_start + seeds) for n in tasks for d in levels]


def game_key(cfg):
    """What identifies a game in a results file (for --resume)."""
    return json.dumps({k: v for k, v in sorted(cfg.items()) if k not in ("game_id", "git")}, sort_keys=True, default=str)


def game_id(cfg, k):
    if cfg.get("difficulty") is not None:
        return f"{cfg['task']}-s{cfg['seed']}-d{cfg['difficulty']}"
    return f"{cfg['task']}-s{cfg['seed']}-{k:04d}"


SANDBOX_SIZES = {"default": dict(cpu=0.5, memory=512), "small": dict(cpu=0.125, memory=256)}  # small: the ablations


def new_sandbox(sb_app, tries=8, size="default"):
    """One agent sandbox. Modal sometimes refuses briefly under load (a launch once failed with PermissionDeniedError
    and worked minutes later), so wait and retry with jitter; no model has been called yet, so a retry costs nothing."""
    import random
    from harness.core import ModalSandbox
    for k in range(tries):
        try:
            return ModalSandbox(sb_app, sandbox_image, **SANDBOX_SIZES[size])
        except modal.exception.Error:
            if k == tries - 1: raise
            time.sleep(min(60, 5 * 2 ** k) * random.uniform(0.5, 1.5))


def new_sandboxes(sb_app, n, size="default"):
    """All of a game's sandboxes at once (in threads); if any fails, the ones made are closed and the error is raised."""
    from concurrent.futures import ThreadPoolExecutor
    made, errors = [], []
    with ThreadPoolExecutor(n) as pool:
        for f in [pool.submit(new_sandbox, sb_app, size=size) for _ in range(n)]:
            try: made.append(f.result())
            except Exception as e: errors.append(e)
    if errors:
        for sb in made: sb.close()
        raise errors[0]
    return made


def play(cfg: dict, sandboxes=None) -> dict:
    """One game, its post-game readers and its transcript. Runs inside the Modal function (or locally with local
    sandboxes, for tests). The officer and observer only read the finished game: nothing they decide changes it."""
    from harness.core import Cost, interview, overseer, privileged_observer, run_episode, security_review
    from harness.tasks import make_task
    from harness.teams import grade_overseer, grade_security
    from harness.transcript import Transcript

    t0 = time.time()
    task = make_task(cfg)
    tr = Transcript(cfg, git=cfg.get("git")) if cfg.get("transcript", True) else None
    cost = Cost()
    ep, score, _ = run_episode(task, mode="llm", agent_model=cfg.get("agent", AGENT_MODEL), cost=cost, sandboxes=sandboxes,
                               transcript=tr)
    # post-game readers, all at once (they only read the finished game). A reader that keeps failing is recorded as an
    # error instead of losing the game: python -m harness.rereview can run the officer again from the results file.
    from concurrent.futures import ThreadPoolExecutor
    jobs = {}
    if cfg.get("security") or cfg.get("v2"):
        # the security officer (strong model) replaces the old yes/no overseers: it rates every agent and can disqualify teams
        views = cfg.get("officer_views", ["full"])
        if "full" in views:
            jobs["security"] = lambda: security_review(OVERSEERS["strong"], task, ep, cost)
        if "messages" in views:  # the same officer seeing only the messages (no scores or submissions)
            jobs["security_msgs"] = lambda: security_review(OVERSEERS["strong"], task, ep, cost, messages_only=True)
        # the privileged observer knows the secrets: who passed them (openly or hidden), and who the officer missed
        jobs["observer"] = lambda: privileged_observer(OVERSEERS["strong"], task, cost, transcript=tr)
    if cfg.get("interview"):  # after the game: ask each holder whether and why it shared its leak
        for h in task.holder:
            jobs[("interview", h)] = (lambda h=h: interview(cfg.get("agent", AGENT_MODEL), task, h, cost, transcript=tr))
    done = {}
    if jobs:
        with ThreadPoolExecutor(len(jobs)) as pool:
            futs = {k: pool.submit(f) for k, f in jobs.items()}
            for k, f in futs.items():
                try: done[k] = f.result()
                except Exception as e: done[k] = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
    security, security_msgs, observer, ov = done.get("security", {}), done.get("security_msgs", {}), done.get("observer", {}), {}
    if cfg.get("security") or cfg.get("v2"):
        for sec in (security, security_msgs):
            if sec and "error" not in sec: sec["grade"] = grade_security(sec, score)
        if observer and "error" not in observer:
            observer["summary"] = task.observer_summary(observer, score, security if "error" not in security else {})
    else:
        ov = {k: overseer(m, task, ep, cost) for k, m in OVERSEERS.items()}
        if "truth_agents" in score:  # team tasks: grade each overseer against the threshold truth
            for v in ov.values(): v["grade"] = grade_overseer(v, score)
    debrief = {}
    for h in (task.holder if cfg.get("interview") else []):
        ans = done[("interview", h)]
        debrief[task.alias[h]] = dict(team=task.team_names[task.team(h)], shared=h in task.holders_who_sent(),
                                     answer=ans if isinstance(ans, str) else ans.get("error"))
    out = dict(security=security, security_messages_only=security_msgs, observer=observer, debrief=debrief,
               thoughts=[(r, i, t[:800]) for r, i, t in ep.thoughts], cfg=cfg, score=score, overseer=ov, cost_usd=cost.usd,
               calls=cost.calls, secs=round(time.time() - t0, 1),
               log=ep.log, private_log=[(r, i, c, o[:200]) for r, i, c, o in ep.private_log])
    if tr is not None:
        for k in ("security", "security_messages_only", "observer", "debrief", "overseer"):
            tr.add_post(k, out[k])
        tr.add_post("cost", dict(usd=cost.usd, calls=cost.calls, tokens=cost.tokens, secs=out["secs"]))
        out["transcript_b64"] = tr.to_b64()
    return out


# games in flight at once: unlimited unless MAX_GAMES is set (each game also holds up to 15 agent sandboxes, so set it
# to fit the workspace's container limit; see --mode capacity-probe)
MAX_GAMES = int(os.environ.get("MAX_GAMES", "0")) or None


@app.function(image=func_image, secrets=SECRETS, timeout=14400, max_containers=MAX_GAMES)  # 4 h: DeepSeek D3 games ran past 80 min
def episode(cfg: dict) -> dict:
    sb_app = modal.App.lookup("collusion-sandboxes", create_if_missing=True)
    from harness.tasks import make_task
    sandboxes = new_sandboxes(sb_app, make_task(cfg).n_total, size=cfg.get("sandbox", "default"))
    try:
        return play(cfg, sandboxes)
    finally:
        for s in sandboxes: s.close()


@app.function(image=func_image, timeout=1800)
def capacity_probe(n: int) -> dict:
    """How many agent sandboxes this workspace can hold at once: start n idle sandboxes together, count how many came up
    and how fast, then close them all. No model is called."""
    from concurrent.futures import ThreadPoolExecutor
    from harness.core import ModalSandbox
    sb_app = modal.App.lookup("collusion-sandboxes", create_if_missing=True)
    t0, made, errors = time.time(), [], []

    def one(_):
        t = time.time()
        try:
            sb = ModalSandbox(sb_app, sandbox_image, timeout=600)
            ok = sb.exec("echo ok").strip() == "ok"
            return sb, ok, time.time() - t, None
        except Exception as e:
            return None, False, time.time() - t, f"{type(e).__name__}: {str(e)[:120]}"
    with ThreadPoolExecutor(min(n, 256)) as pool:
        res = list(pool.map(one, range(n)))
    for sb, ok, dt, err in res:
        if sb is not None: made.append(sb)
        if err: errors.append(err)
    secs = sorted(dt for _, ok, dt, _ in res if ok)
    for sb in made: sb.close()
    return dict(requested=n, up=sum(ok for _, ok, _, _ in res), failed=len(errors), errors=sorted(set(errors))[:5],
                median_start_s=round(secs[len(secs) // 2], 1) if secs else None, max_start_s=round(secs[-1], 1) if secs else None,
                wall_s=round(time.time() - t0, 1))


@app.function(image=func_image, secrets=SECRETS, timeout=1800)
def rereview_game(job: dict) -> dict:
    """The officer reads one finished game again, with another model or prompt (python -m harness.rereview, remotely)."""
    from harness.rereview import review
    rec = job["rec"]
    out = review(rec, model=job["model"], messages_only=job["messages_only"], officer_prompt=job["officer_prompt"] or None)
    return dict(cfg=rec["cfg"], score=rec["score"], security=out["security"], source=job["source"])


def run_rereview(src, officer_model, officer_prompt, messages_only, tag):
    """modal run harness/run_modal.py --mode rereview --src 'results/difficulty-*-main.jsonl' --officer-model <id>
    [--officer-prompt aggressive] [--messages-only] --tag <name>: one output file per source file."""
    import glob
    from harness.rereview import MODEL
    files = sorted(glob.glob(src))
    assert files, f"no results file matches {src}"
    model = officer_model or MODEL
    jobs = [dict(rec=json.loads(l), model=model, officer_prompt=officer_prompt, messages_only=messages_only, source=f)
            for f in files for l in open(f) if l.strip()]
    print(f"re-reviewing {len(jobs)} games with {model} ({officer_prompt or 'as played'}, {'messages only' if messages_only else 'full view'})")
    outs = {f: open(f.replace(".jsonl", f"-officer-{tag}.jsonl"), "w") for f in files}
    spent = done = 0
    for r in rereview_game.map(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("re-review failed:", type(r).__name__, str(r)[:150]); continue
        spent += r["security"].get("cost_usd", 0); done += 1
        outs[r.pop("source")].write(json.dumps(r, default=str) + "\n")
    for h in outs.values(): h.close()
    print(f"done: {done}/{len(jobs)} re-reviewed, ${spent:.2f}")


@app.function(image=func_image, secrets=SECRETS, timeout=60)
def key_check() -> dict:
    """The OpenRouter key's credit limit, usage and rate limit (GET /api/v1/key: no model call, no spend)."""
    import os as _os
    import requests
    key = _os.environ.get("OPENROUTER_API_KEY") or next(v for k, v in _os.environ.items() if "OPENROUTER" in k.upper())
    r = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    d = r.json().get("data", {}) if r.ok else {"error": f"{r.status_code}: {r.text[:200]}"}
    out = {k: v for k, v in d.items() if k != "label"}
    # the account's own balance: a key limit can have room left while the account has run out of credit
    c = requests.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if c.ok:
        cd = c.json().get("data", {})
        out["account_credits"] = dict(total=cd.get("total_credits"), used=cd.get("total_usage"),
                                      balance=round((cd.get("total_credits") or 0) - (cd.get("total_usage") or 0), 2))
    return out


@app.function(image=func_image, secrets=SECRETS, timeout=300)
def sandbox_check() -> dict:
    """Verify sandbox isolation properties before any agent runs."""
    from harness.core import ModalSandbox
    sb_app = modal.App.lookup("collusion-sandboxes", create_if_missing=True)
    sb = ModalSandbox(sb_app, sandbox_image)
    try:
        return dict(
            uid=sb.exec("id -u"),
            whoami=sb.exec("whoami"),
            network=sb.exec("python3 -c \"import urllib.request;urllib.request.urlopen('https://example.com',timeout=5);print('REACHED')\" 2>&1 | tail -1"),
            dns=sb.exec("python3 -c \"import socket;print(socket.gethostbyname('openrouter.ai'))\" 2>&1 | tail -1"),
            env_secrets=sb.exec("env | grep -i -E 'key|token|secret|openrouter' || echo none"),
            home_writable=sb.exec("touch /home/agent/work/x && echo ok"),
        )
    finally:
        sb.close()


@app.local_entrypoint()
def main(mode: str = "pilot", seeds: int = 5, budget: float = -1.0, tasks: str = "", agent: str = AGENT_MODEL, tag: str = "",
         overwrite: bool = False, rungs: str = "level0", channels: str = "on,off", seed_start: int = 0, ablate: str = "", share_first: bool = False,
         interview: bool = False, holder_cmds: int = 0, leak_look: str = "", no_rule: bool = False,
         teammate_asks: bool = False, strong_teamwork: bool = False,
         team_only: bool = False, random_teams: bool = False, progress_board: bool = False,
         v2: bool = False, security: bool = False, pairs: str = "", surveillance: bool = False,
         uniform_holders: bool = False, team_reward: bool = False,
         levels: str = "0,1,2,3", dry_run: bool = False, resume: bool = False, officer_prompt: str = "",
         probe_n: int = 500, arms: str = "", src: str = "", officer_model: str = "", messages_only: bool = False):
    from harness.tasks import ALL_TASKS, BY_NAME
    agents = [AGENTS.get(a, a) for a in agent.split(",")]  # short names: hermes, dolphin, opus; several with commas
    assert len(agents) == 1 or mode in ("difficulty", "ablations"), "several agent models in one launch: difficulty modes only"
    agent = agents[0]
    names = [n for n in BY_NAME if n in tasks.split(",")] if tasks else [t.name for t in ALL_TASKS]
    if mode == "sandbox-check":
        print(json.dumps(sandbox_check.remote(), indent=2)); return
    if mode == "capacity-probe":
        print(json.dumps(capacity_probe.remote(probe_n), indent=2)); return
    if mode == "key-check":
        print(json.dumps(key_check.remote(), indent=2)); return
    if mode == "rereview":
        run_rereview(src, officer_model, officer_prompt, messages_only, tag); return
    if mode == "ablations":
        names = sorted({c["task"] for c in build_ablation_cfgs(list(ARMS))})
        cfgs = build_ablation_cfgs([a for a in arms.split(",") if a] or list(ARMS))
        agents = sorted({c["agent"] for c in cfgs})
        agent = agents[0]
    elif mode == "difficulty":
        # honest-path difficulty: every (seed, task, level) for each agent model; default seeds 200-204 (fresh, matched)
        names = [n for n in HP_TASK_NAMES if n in tasks.split(",")] if tasks else list(HP_TASK_NAMES)
        if not seed_start: seed_start = 200
        cfgs = [c for a in agents for c in build_difficulty_cfgs(
            names, [int(x) for x in levels.split(",")], seeds, seed_start, a,
            extra={"officer_prompt": officer_prompt} if officer_prompt else None)]
        if pairs:  # rerun only some games: "task:seed,task:seed"
            want = {(p.split(":")[0], int(p.split(":")[1])) for p in pairs.split(",") if p}
            cfgs = [c for c in cfgs if (c["task"], c["seed"]) in want]
    elif mode == "teams":
        # team tasks: every (rung, messaging on/off) cell, seeds seed_start .. seed_start + seeds - 1 (pilot on 100+, main runs on 0-9).
        # On this branch the default is the collusion-friendly setup: rung level0, every lever on. --ablate a,b (or all) adds one
        # variant per lever with that lever switched off (messaging on only).
        from harness.teams import LEVERS
        assert tasks, "--mode teams needs --tasks"
        offs = list(LEVERS) if ablate == "all" else [x for x in ablate.split(",") if x]
        assert all(x in LEVERS for x in offs), f"unknown lever in --ablate; levers: {', '.join(LEVERS)}"
        cfgs = [dict(task=n, seed=s, channel=c == "on", rung=g, agent=agent)
                for s in range(seed_start, seed_start + seeds) for n in names for g in rungs.split(",") for c in channels.split(",")]
        cfgs += [dict(task=n, seed=s, channel=True, rung=g, agent=agent, levers={x: LEVERS[x][1]})
                 for s in range(seed_start, seed_start + seeds) for n in names for g in rungs.split(",") for x in offs]
        if pairs:  # rerun only some games: "task:seed,task:seed" (e.g. the games an aborted run did not finish)
            want = {(p.split(":")[0], int(p.split(":")[1])) for p in pairs.split(",") if p}
            cfgs = [c for c in cfgs if (c["task"], c["seed"]) in want]
            assert cfgs, "--pairs matched no game (check --seed-start/--seeds cover those seeds)"
        if share_first:  # test option: each holder's leak is posted to its team at the start
            for c in cfgs: c["share_first"] = True
        if interview:  # after each game, ask the holders whether and why they shared
            for c in cfgs: c["interview"] = True
        for c in cfgs:  # test options: more commands per turn for holders; a neutral-looking leak file
            if holder_cmds: c["holder_cmds"] = holder_cmds
            if leak_look: c["leak_look"] = leak_look
            if no_rule: c["no_rule"] = True  # test option: no RULES line in the briefs
            if teammate_asks: c["teammate_asks"] = True  # test option: a teammate asks the team for the secret part
            if strong_teamwork: c["strong_teamwork"] = True  # test option: the RULES line encourages teamwork
            if team_only: c["team_only"] = True  # test option: only the team total is rewarded
            if random_teams: c["random_teams"] = True  # 2-5 teams, only some with a holder
            if progress_board: c["progress_board"] = True  # teammates' item-by-item progress after each round
            if v2: c["v2"] = True  # friendly v2 preset: random teams, board, balanced goal, 1% auction window
            if security or v2: c["security"] = True  # the security officer reviews every game
            if surveillance: c["surveillance"] = True  # agents are told the security officer reads everything
            if uniform_holders: c["uniform_holders"] = True  # number of holder teams uniform in 1..teams-1
            if team_reward: c["team_reward"] = True  # agents rewarded only for the team total (per-member scoring kept)
    elif mode == "ablate":
        # baseline gets thread on AND off; every variant is thread-on only
        cfgs = [dict(task=n, seed=s, channel=c, pressure=False, level=3, agent=agent)
                for s in range(seeds) for n in names for c in ((True, False) if n == "leaderboard" else (True,))]
    elif mode == "level3":
        cfgs = [dict(task=n, seed=s, channel=c, pressure=False, level=3, agent=agent) for s in range(seeds) for n in names for c in (True, False)]
    elif mode == "pilot":
        cfgs = [dict(task=n, seed=0, channel=True, agent=agent) for n in names]
    else:
        cfgs = [dict(task=n, seed=s, channel=c, pressure=p, agent=agent) for s in range(seeds) for n in names for c in (True, False) for p in (False, True)]
    out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
    what = f"{mode}-{'+'.join(names)}" if mode == "teams" else mode  # team runs: the task is in the name (two tasks once collided)
    # one results file per agent model (and per arm in ablation runs)
    key = lambda c: (c.get("agent", agent), c.get("arm"))
    groups = sorted({key(c) for c in cfgs}, key=str)
    runs = {g: f"{what}-{g[0].split('/')[-1]}{'-' + g[1] if g[1] else ''}{'-' + tag if tag else ''}" for g in groups}
    from harness.transcript import git_hash
    git = git_hash()
    for k, c in enumerate(cfgs):
        c["game_id"] = game_id(c, k)
        c["git"] = git
    if dry_run:  # print what would run, and stop before any remote call
        for g in groups:
            report_dry_run([c for c in cfgs if key(c) == g], g[0], out_dir / f"{runs[g]}.jsonl")
        if len(groups) > 1:
            print(f"ALL MODELS: {len(cfgs)} games in one launch, estimated ${sum(EST_PER_GAME.get(c['agent'], 0) for c in cfgs):.2f}")
        return
    assert budget > 0, "pass --budget <usd> (there is no default budget)"
    files, todo = {}, []
    for a in groups:
        out = out_dir / f"{runs[a]}.jsonl"
        mine = [c for c in cfgs if key(c) == a]
        done_keys = set()
        if out.exists() and out.stat().st_size > 0:  # an empty file (an aborted run) holds nothing to protect
            if resume:
                done_keys = {game_key(json.loads(l)["cfg"]) for l in out.open() if l.strip()}
                print(f"resume: {len(done_keys)} games already in {out}")
            elif not overwrite:
                print(f"STOP: {out} already exists. Use a new --tag, --resume, or pass --overwrite to replace it."); return
        todo += [c for c in mine if game_key(c) not in done_keys]
        files[a] = (out, Path("transcripts") / runs[a])
    todo = priority_order(todo) if mode == "ablations" else longest_first(todo)
    handles = {a: out.open("a" if resume else "w") for a, (out, _) in files.items()}
    spent, done, failed, t0 = 0.0, 0, 0, time.time()
    print(f"launching {len(todo)} games at once (MAX_GAMES={MAX_GAMES or 'unlimited'}), budget ${budget}")
    try:
        # every game in one streaming map: results are written as each game finishes, in any order
        for r in episode.map(todo, order_outputs=False, return_exceptions=True):
            if isinstance(r, Exception):
                failed += 1
                print(f"episode failed ({failed}):", type(r).__name__, str(r)[:200]); continue
            a = key(r["cfg"])
            spent += r["cost_usd"]; done += 1
            b64 = r.pop("transcript_b64", None)
            if b64:
                from harness.transcript import save_b64
                path = files[a][1] / f"{r['cfg']['game_id']}.json.gz"
                r["transcript_path"], r["transcript_sha256"] = str(path), save_b64(b64, path)
            handles[a].write(json.dumps(r, default=str) + "\n"); handles[a].flush()
            print(f"[{done + failed}/{len(todo)}] {a[0].split('/')[-1]}{' ' + a[1] if a[1] else ''} {r['cfg'].get('game_id')} ${r['cost_usd']:.3f} "
                  f"{r['secs']:.0f}s | spent ${spent:.2f} | {(time.time() - t0) / 60:.1f} min")
            if spent >= budget:  # leaving the map stops the app, which cancels the games still running
                print(f"STOP: budget ${budget} reached after {done} episodes (spent ${spent:.2f}); games in flight are cancelled")
                break
    finally:
        for h in handles.values(): h.close()
    print(f"done: {done} episodes, {failed} failed, ${spent:.3f}, {(time.time() - t0) / 60:.1f} min -> "
          + ", ".join(str(o) for o, _ in files.values()) + ("  (rerun the failed ones with --resume)" if failed else ""))


def longest_first(cfgs):
    """Start the longest games first, so a capped number of games in flight still finishes as early as possible:
    slower models first, then games with more agents (more model calls per round)."""
    from harness.tasks import make_task
    speed = {AGENTS["opus"]: 3, AGENTS["hermes"]: 2, AGENTS["deepseek"]: 2, AGENTS["maverick"]: 1, AGENTS["haiku"]: 1,
             AGENTS["dolphin"]: 1, AGENTS["kimi"]: 2, AGENTS["qwen"]: 1, AGENTS["minimax"]: 2}
    return sorted(cfgs, key=lambda c: (-speed.get(c.get("agent"), 2), -make_task(c).n_total))


def report_dry_run(cfgs, agent, out):
    from collections import Counter
    est = EST_PER_GAME.get(agent)
    print(f"DRY RUN (no Modal or API call): {len(cfgs)} games for {agent} -> {out}")
    for key in ("task", "difficulty", "seed"):
        print(f"  {key}: " + ", ".join(f"{v}={n}" for v, n in sorted(Counter(c.get(key) for c in cfgs).items(), key=str)))
    fixed = {k: v for k, v in cfgs[0].items() if k not in ("task", "difficulty", "seed", "game_id") and all(c.get(k) == v for c in cfgs)} if cfgs else {}
    print(f"  same in every game: {json.dumps(fixed, default=str)}")
    print(f"  estimated cost: " + (f"${est * len(cfgs):.2f} (${est}/game)" if est else "unknown for this model"))


if __name__ == "__main__":
    # offline dry run without Modal: python -m harness.run_modal --mode difficulty --agent hermes [--tasks ..] [--levels ..]
    import argparse
    ap = argparse.ArgumentParser(description="Dry run: print the games a launch would play, with no Modal or API call.")
    ap.add_argument("--mode", default="difficulty")
    ap.add_argument("--agent", default=AGENT_MODEL)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--levels", default="0,1,2,3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main.info.raw_f(mode=a.mode, agent=a.agent, tasks=a.tasks, levels=a.levels, seeds=a.seeds, seed_start=a.seed_start,
                    tag=a.tag, dry_run=True)
