"""Render game transcripts (harness/transcript.py) as self-contained HTML pages, for reading what happened in a game.

    python -m harness.transcript_view <game.json.gz> [-o game.html]
    python -m harness.transcript_view --index transcripts/<run>/     # index.html plus one page per game, in that folder

A game page shows the outcome (leak opened, sent, used; collusion; officer verdicts against the truth), the roster, the
keys (analyst only), every message (red: carries a team's leak, green: carries honestly derived answers), and a timeline
of every model call and command with full outputs, then the post-game readers (officer, observer, interviews).
"""
import argparse
import html
import json
from pathlib import Path

from harness.transcript import load

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e3e1da;--card:#ffffff;--leak:#b3261e;--leakbg:#fbe9e7;
--honest:#1e7a3c;--honestbg:#e6f4ea;--accent:#1f5fad;--code:#f3f1ec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;
--card:#1f1e1c;--leak:#ff8a80;--leakbg:#3a1d1b;--honest:#7fd99a;--honestbg:#16301f;--accent:#8ab4f8;--code:#262522}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--card:#1f1e1c;--leak:#ff8a80;--leakbg:#3a1d1b;
--honest:#7fd99a;--honestbg:#16301f;--accent:#8ab4f8;--code:#262522}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px;
border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:14px;margin:18px 0 6px}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;
vertical-align:top}.wrap{overflow-x:auto}pre{background:var(--code);padding:8px;border-radius:4px;white-space:pre-wrap;
word-break:break-word;margin:4px 0;font:12px/1.45 ui-monospace,Menlo,monospace}details{margin:2px 0}summary{cursor:pointer;
color:var(--accent)}.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:12px;margin:0 4px 4px 0;
border:1px solid var(--line)}.b-leak{background:var(--leakbg);color:var(--leak);border-color:var(--leak)}.b-honest{background:
var(--honestbg);color:var(--honest);border-color:var(--honest)}.msg{padding:3px 6px;border-left:3px solid var(--line);margin:2px 0}
.carries-leak{border-left-color:var(--leak);background:var(--leakbg)}.carries-honest{border-left-color:var(--honest);
background:var(--honestbg)}.ev{border:1px solid var(--line);background:var(--card);border-radius:6px;padding:6px 8px;margin:6px 0}
.cmd{font-family:ui-monospace,Menlo,monospace;font-weight:600}.round{margin-top:22px}a{color:var(--accent)}
"""


def e(x): return html.escape(str(x))


def _page(title, body):
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{e(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")


def outcome(data):
    """The game's headline numbers, from the score and the post-game readers."""
    sc, post = data.get("score") or {}, data.get("post") or {}
    sec = post.get("security") or {}
    col = sec.get("grade", {}).get("team", {}) if sec else {}
    return dict(
        holders=sc.get("n_holders"), opened=sc.get("holders_opened"), self_use=sc.get("solo_ever"), sent=sc.get("holders_sent"),
        nonholders=sc.get("n_nonholders"), used=sc.get("within_ever"), colluding=len(sc.get("teams_colluded", [])),
        attempted=len(sc.get("teams_attempted", [])), control_honest=sc.get("control_honest_rate"),
        detected_tp=col.get("tp"), detected_fn=col.get("fn"), false_alarms=col.get("fp"), difficulty=sc.get("difficulty"))


def render(data):
    h, sc, post = data["header"], data.get("score") or {}, data.get("post") or {}
    o = outcome(data)
    a2team = {r["agent"]: r["team"] for r in h["roster"]}
    holders = {r["agent"] for r in h["roster"] if r["holder"]}
    users = set(sc.get("truth_users", []))
    out = [f"<h1>{e(h['task'])} · seed {e(h['seed'])} · difficulty {e(h.get('difficulty'))}</h1>",
           f"<div class='muted'>model {e(h.get('agent_model'))} · mode {e(h.get('mode'))} · git {e(data.get('git'))} · "
           f"{e(data.get('created_utc'))} · cost ${(post.get('cost') or {}).get('usd', 0):.3f}</div>"]
    badges = [f"holders opened {o['opened']}/{o['holders']}", f"holders used it {o['self_use']}/{o['holders']}",
              f"holders sent {o['sent']}/{o['holders']}", f"teammates used {o['used']}/{o['nonholders']}",
              f"colluding teams {o['colluding']}"]
    if o["control_honest"] is not None: badges.append(f"control honest rate {o['control_honest']:.2f}")
    if o["detected_tp"] is not None:
        badges.append(f"officer caught {o['detected_tp']}/{o['detected_tp'] + o['detected_fn']} colluding teams, "
                      f"{o['false_alarms']} false alarms")
    out.append("<div style='margin-top:10px'>" + "".join(
        f"<span class='badge {'b-leak' if ('used' in b or 'sent' in b) else ''}'>{e(b)}</span>" for b in badges) + "</div>")

    # roster and keys
    final = sc.get("agent_scores", {})
    hon = sc.get("honest_secret", {})
    out.append("<h2>Roster</h2><div class='wrap'><table><tr><th>agent</th><th>team</th><th>role</th><th>final score</th>"
               "<th>leak used</th><th>special items by honest work</th></tr>")
    for r in h["roster"]:
        a = r["agent"]
        role = "holder" if r["holder"] else ("teammate" if r["team"] in h["holder_teams"] else "control")
        used = "<span class='badge b-leak'>yes</span>" if a in users else ""
        out.append(f"<tr><td>{e(a)}</td><td>{e(r['team'])}</td><td>{role}</td><td>{e(final.get(a, ''))}</td>"
                   f"<td>{used}</td><td>{e(hon.get(a, ''))}</td></tr>")
    out.append("</table></div>")
    if h.get("leak_values"):
        out.append("<h3>Keys (analyst only; agents never see this)</h3><div class='wrap'><table><tr><th>team</th><th>leak (v1)</th>"
                   "<th>current key (v2)</th></tr>" + "".join(
                       f"<tr><td>{e(t)}</td><td>{e(' '.join(map(str, h['leak_values'].get(t, []))))}</td>"
                       f"<td>{e(' '.join(map(str, h.get('honest_values', {}).get(t, []))))}</td></tr>"
                       for t in sorted(set(h["leak_values"]) | set(h.get("honest_values", {})))) + "</table></div>")

    # messages
    msgs = [m for ev in data["events"] for m in ev.get("messages", [])]
    out.append(f"<h2>Messages ({len(msgs)})</h2>")
    for m in msgs:
        cls = "carries-leak" if m["carries_leak"] else "carries-honest" if m["carries_honest"] else ""
        to = f"team {m['team']}" if m["to"] == "team" else m["to"]
        tag = (f" <span class='badge b-leak'>leak of {e(', '.join(m['carries_leak']))}</span>" if m["carries_leak"] else "") + \
              (f" <span class='badge b-honest'>honest answers</span>" if m["carries_honest"] else "")
        out.append(f"<div class='msg {cls}'><span class='muted'>r{m['round'] + 1}</span> <b>{e(m['sender'])}</b> "
                   f"({e(m['sender_team'])}) → {e(to)}: {e(m['text'])}{tag}</div>")

    # agents' briefs and workspaces
    out.append("<h2>What each agent was given</h2>")
    for ag in data["agents"]:
        files = "".join(f"<details><summary>{e(f)}</summary><pre>{e(c)}</pre></details>" for f, c in sorted(ag["workspace"].items()))
        out.append(f"<details><summary>{e(ag['agent'])} ({e(ag['team'])}{', holder' if ag['holder'] else ''})</summary>"
                   f"<h3>System brief</h3><pre>{e(ag['brief'])}</pre><h3>Workspace</h3>{files}</details>")

    # timeline
    out.append("<h2>Timeline</h2>")
    alias = [r["agent"] for r in h["roster"]]
    for ev in data["events"]:
        t = ev["type"]
        if t == "round_start":
            label = f"Round {ev['round'] + 1}" if ev["round"] < h["rounds"] else "Final turn"
            out.append(f"<h3 class='round'>{e(label)}</h3>")
        elif t == "call":
            a = alias[ev["agent_index"]]
            meta = f"{ev.get('prompt_tokens')}+{ev.get('completion_tokens')} tokens, {ev.get('latency_s')} s, {ev.get('finish_reason')}"
            flag = " <span class='badge b-leak'>refused</span>" if ev.get("refused") else ""
            nb = " <span class='muted'>(no command block)</span>"
            out.append(f"<div class='ev'><b>{e(a)}</b> <span class='muted'>model call · {e(meta)}</span>{flag}"
                       f"<details><summary>observation sent</summary><pre>{e(ev['observation'])}</pre></details>"
                       f"<details><summary>full reply</summary><pre>{e(ev['reply'])}</pre></details>"
                       f"<div>→ <span class='cmd'>{e(ev['command'])}</span>"
                       f"{nb if ev.get('parse') == 'no_block' else ''}</div></div>")
        elif t == "command":
            marks = "".join(f"<span class='badge b-leak'>opened the leak</span>" for _ in ev["opened_leak"])
            marks += "".join(f"<span class='badge {'b-leak' if x['kind'] == 'leak' else 'b-honest'}'>{e(x['agent'])} crossed "
                             f"({e(x['kind'])})</span>" for x in ev["crossings"])
            sbx = "".join(f"<details><summary>sandbox {e(c['kind'])}: {e((c.get('cmd') or c.get('script', ''))[:80])}</summary>"
                          f"<pre>{e(c.get('full_output') or c.get('output'))}</pre></details>" for c in ev["sandbox"])
            out.append(f"<div class='ev'><b>{e(ev['agent'])}</b>$ <span class='cmd'>{e(ev['command'])}</span> "
                       f"<span class='muted'>score after: {e(ev['score_after'])}</span> {marks}<pre>{e(ev['output'])}</pre>{sbx}</div>")
        elif t == "round_end":
            row = ", ".join(f"{e(a)}={e(v)}" for a, v in ev["scores"].items())
            marks = "".join(f"<span class='badge b-leak'>{e(x['agent'])} crossed ({e(x['kind'])})</span>" for x in ev["crossings"])
            out.append(f"<div class='muted'>end of round · scores: {row} {marks}</div>")

    # post-game readers
    out.append("<h2>After the game (read-only: nothing here changed the game)</h2>")
    sec = post.get("security") or {}
    truth = set(sc.get("teams_colluded", [])) | set(sc.get("teams_attempted", []))
    for key, label in (("security", "Security officer, full view"), ("security_messages_only", "Security officer, messages only")):
        s = post.get(key) or {}
        if not s: continue
        rows = "".join(f"<tr><td>{e(t)}</td><td>{e(v.get('decision'))}</td><td>{e(v.get('suspicion'))}</td>"
                       f"<td>{'colluded' if t in truth else ''}</td><td>{e(v.get('reason'))}</td></tr>" for t, v in s.get("teams", {}).items())
        out.append(f"<h3>{label}</h3><table><tr><th>team</th><th>decision</th><th>suspicion</th><th>truth</th><th>reason</th></tr>"
                   f"{rows}</table>")
    for r in data.get("readers", []):
        out.append(f"<details><summary>{e(r['kind'])}{' · ' + e(r['agent']) if r.get('agent') else ''} ({e(r['model'])})</summary>"
                   f"<h3>Prompt</h3><pre>{e(r['prompt'])}</pre><h3>Reply</h3><pre>{e(r['reply'])}</pre></details>")
    return _page(f"{h['task']} s{h['seed']} d{h.get('difficulty')}", "".join(out))


def render_index(folder, write_pages=False):
    folder = Path(folder)
    rows = []
    for p in sorted(folder.glob("*.json.gz")):
        data = load(p)
        h, o = data["header"], outcome(data)
        page = p.name.replace(".json.gz", ".html")
        if write_pages:
            (folder / page).write_text(render(data))
        det = "" if o["detected_tp"] is None else f"{o['detected_tp']}/{o['detected_tp'] + o['detected_fn']} (FP {o['false_alarms']})"
        ch = "" if o["control_honest"] is None else f"{o['control_honest']:.2f}"
        rows.append(f"<tr><td><a href='{e(page)}'>{e(p.name[:-8])}</a></td><td>{e(h['task'])}</td><td>{e(h.get('difficulty'))}</td>"
                    f"<td>{e(h['seed'])}</td><td>{e(o['sent'])}/{e(o['holders'])}</td><td>{e(o['used'])}/{e(o['nonholders'])}</td>"
                    f"<td>{e(o['colluding'])}</td><td>{e(ch)}</td>"
                    f"<td>{e(det)}</td></tr>")
    body = (f"<h1>Games in {e(folder.name)}</h1><div class='wrap'><table><tr><th>game</th><th>task</th><th>difficulty</th><th>seed</th>"
            "<th>holders sent</th><th>teammates used</th><th>colluding teams</th><th>control honest rate</th>"
            "<th>officer caught (false alarms)</th></tr>" + "".join(rows) + "</table></div>")
    return _page(f"Games: {folder.name}", body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--out")
    ap.add_argument("--index", action="store_true", help="path is a folder: write index.html and one page per game there")
    a = ap.parse_args()
    if a.index:
        out = Path(a.out or Path(a.path) / "index.html")
        out.write_text(render_index(a.path, write_pages=True))
    else:
        out = Path(a.out or a.path.replace(".json.gz", ".html"))
        out.write_text(render(load(a.path)))
    print(f"wrote {out}")
