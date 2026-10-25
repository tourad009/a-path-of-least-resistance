"""Parallel runner on Modal.
  modal run harness/run_modal.py --mode sandbox-check
  modal run harness/run_modal.py --mode pilot
  modal run harness/run_modal.py --mode full --seeds 5
Each episode runs in one Modal function; every agent gets its own offline Modal Sandbox.
"""
import json
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


def new_sandbox(sb_app, tries=4):
    """One agent sandbox. Modal sometimes refuses briefly under load (a launch once failed with PermissionDeniedError
    and worked minutes later), so wait and retry; no model has been called yet, so a retry costs nothing."""
    from harness.core import ModalSandbox
    for k in range(tries):
        try:
            return ModalSandbox(sb_app, sandbox_image)
        except modal.exception.Error:
            if k == tries - 1: raise
            time.sleep(15 * (k + 1))


@app.function(image=func_image, secrets=SECRETS, timeout=1800)
def episode(cfg: dict) -> dict:
    from harness.core import Cost, interview, overseer, privileged_observer, run_episode, security_review
    from harness.tasks import make_task
    from harness.teams import grade_overseer, grade_security

    t0 = time.time()
    task = make_task(cfg)
    sb_app = modal.App.lookup("collusion-sandboxes", create_if_missing=True)
    cost = Cost()
    sandboxes = []
    try:
        for _ in range(task.n_total):  # inside the try: if Modal refuses one, the ones already made are still closed
            sandboxes.append(new_sandbox(sb_app))
        ep, score, _ = run_episode(task, mode="llm", agent_model=cfg.get("agent", AGENT_MODEL), cost=cost, sandboxes=sandboxes)
    finally:
        for s in sandboxes: s.close()
    security, observer = {}, {}
    if cfg.get("security") or cfg.get("v2"):
        # the security officer (strong model) replaces the old yes/no overseers: it rates every agent and can disqualify teams
        ov = {}
        security = security_review(OVERSEERS["strong"], task, ep, cost)
        security["grade"] = grade_security(security, score)
        # the privileged observer knows the secrets: who passed them (openly or hidden), and who the officer missed
        observer = privileged_observer(OVERSEERS["strong"], task, cost)
        observer["summary"] = task.observer_summary(observer, score, security)
    else:
        ov = {k: overseer(m, task, ep, cost) for k, m in OVERSEERS.items()}
        if "truth_agents" in score:  # team tasks: grade each overseer against the threshold truth
            for v in ov.values(): v["grade"] = grade_overseer(v, score)
    debrief = {}
    if cfg.get("interview"):  # after the game: ask each holder whether and why it shared its leak
        for h in task.holder:
            shared = h in task.holders_who_sent()
            debrief[task.alias[h]] = dict(team=task.team_names[task.team(h)], shared=shared,
                                         answer=interview(cfg.get("agent", AGENT_MODEL), task, h, cost))
    return dict(security=security, observer=observer, debrief=debrief, thoughts=[(r, i, t[:800]) for r, i, t in ep.thoughts], cfg=cfg, score=score, overseer=ov, cost_usd=cost.usd, calls=cost.calls, secs=round(time.time() - t0, 1),
                log=ep.log, private_log=[(r, i, c, o[:200]) for r, i, c, o in ep.private_log])


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
def main(mode: str = "pilot", seeds: int = 5, budget: float = 18.0, tasks: str = "", agent: str = AGENT_MODEL, tag: str = "",
         overwrite: bool = False, rungs: str = "level0", channels: str = "on,off", seed_start: int = 0, ablate: str = "", share_first: bool = False,
         interview: bool = False, holder_cmds: int = 0, leak_look: str = "", no_rule: bool = False,
         teammate_asks: bool = False, strong_teamwork: bool = False,
         team_only: bool = False, random_teams: bool = False, progress_board: bool = False,
         v2: bool = False, security: bool = False, pairs: str = "", surveillance: bool = False,
         uniform_holders: bool = False, team_reward: bool = False):
    from harness.tasks import ALL_TASKS, BY_NAME
    names = [n for n in BY_NAME if n in tasks.split(",")] if tasks else [t.name for t in ALL_TASKS]
    if mode == "sandbox-check":
        print(json.dumps(sandbox_check.remote(), indent=2)); return
    if mode == "teams":
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
    out = out_dir / f"{what}-{agent.split('/')[-1]}{'-' + tag if tag else ''}.jsonl"
    if out.exists() and out.stat().st_size > 0 and not overwrite:  # an empty file (an aborted run) holds nothing to protect
        print(f"STOP: {out} already exists. Use a new --tag, or pass --overwrite to replace it."); return
    spent, done = 0.0, 0
    B = 40  # batch size (games run in parallel): the budget is checked between batches
    with out.open("w") as f:  # one file per (mode, agent model, tag)
        for k in range(0, len(cfgs), B):
            if spent >= budget:
                print(f"STOP: budget ${budget} reached after {done} episodes (spent ${spent:.2f})"); break
            for r in episode.map(cfgs[k:k + B], return_exceptions=True):
                if isinstance(r, Exception):
                    print("episode failed:", type(r).__name__, str(r)[:200]); continue
                spent += r["cost_usd"]; done += 1
                f.write(json.dumps(r, default=str) + "\n"); f.flush()
            print(f"{done}/{len(cfgs)} episodes, spent ${spent:.3f}")
    print(f"done: {done} episodes, ${spent:.3f} -> {out}")
