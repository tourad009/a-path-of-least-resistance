"""Every number in the paper comes from here: runs the difficulty analysis on the result files and writes numbers.json,
stamped with the git commit and the sha256 of every input file (and checks every game's transcript against its hash).

    python analysis/verify.py results/difficulty-*.jsonl [--out analysis/numbers.json] [--replay]

--replay also replays every transcript and checks it reproduces the game's recorded score.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.analysis_difficulty import analyse, load  # noqa: E402
from harness.transcript import git_hash, load as load_transcript, replay  # noqa: E402


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="analysis/numbers.json")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    recs = load(a.files)
    problems = []
    for r in recs:
        p = r.get("transcript_path")
        if not p or not Path(p).exists():
            problems.append(f"missing transcript: {r['cfg'].get('game_id')}"); continue
        if sha(p) != r.get("transcript_sha256"):
            problems.append(f"transcript hash mismatch: {p}")
        elif a.replay and not replay(load_transcript(p))[1]:
            problems.append(f"transcript does not replay to its score: {p}")
    res = analyse(recs, n_boot=a.boot)
    res["provenance"] = dict(git=git_hash(), inputs={f: sha(f) for f in a.files}, n_games=len(recs),
                             transcript_problems=problems, replayed=a.replay)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    print(f"{len(recs)} games, {len(problems)} transcript problems -> {a.out}")
    for p in problems[:20]: print("  " + p)
