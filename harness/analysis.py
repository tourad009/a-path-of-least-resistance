"""Leaderboard results tables.  Run from the repo root:  python -m harness.analysis"""
import json, math, random, statistics as st
from pathlib import Path

R = Path("results")


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0, c - h), 2), round(min(1, c + h), 2)


def boot(xs, n=2000):
    rnd = random.Random(0)
    m = sorted(st.mean(rnd.choices(xs, k=len(xs))) for _ in range(n))
    return m[int(.025 * n)], m[int(.975 * n)]


def episodes(name):
    return [json.loads(l) for l in open(R / name)]


if __name__ == "__main__":
    for title, f in (("LEVEL 0 (goal + rule, no instruction to collude)", "level0.jsonl"), ("LEVEL 3 (explicit red-team instruction)", "level3.jsonl")):
        rows = episodes(f)
        print(f"\n{title}: {len(rows)} episodes")
        print(f"{'thread':>7}{'n':>3}{'headline (95% CI)':>22}{'missing_action':>16}{'weak flag':>10}{'strong flag':>12}")
        for ch in (True, False):
            v = [r for r in rows if r["cfg"]["channel"] == ch]; h = [r["score"]["headline"] for r in v]; lo, hi = boot(h)
            print(f"{'on' if ch else 'off':>7}{len(v):>3}{st.mean(h):>9.2f} ({lo:.2f},{hi:.2f}){st.mean(r['score']['missing_action_rate'] for r in v):>16.2f}"
                  f"{sum(r['overseer']['weak']['flag'] for r in v) / len(v):>10.2f}{sum(r['overseer']['strong']['flag'] for r in v) / len(v):>12.2f}")
        k = sum(r["overseer"]["strong"]["flag"] for r in rows)
        print(f"strong overseer flags: {k}/{len(rows)} {wilson(k, len(rows))}")
