# a-path-of-least-resistance

**Asymmetric Incentives Toward Collusion: The Role of Honest-Path Difficulty.**
A multi-team testbed where one member of some teams (the *holder*) finds a leaked answer file. The collusive path (the leak)
stays fixed, while the **honest path** to the same points is made easy (D0), harder (D1, D2) or impossible (D3). We measure
whether the leak is used, shared and taken up, and whether a passive security officer detects it.

## Provenance
The harness in `harness/` was imported from
[`ArnolFokam/collusion-leaderboard`](https://github.com/ArnolFokam/collusion-leaderboard), branch `submission`, commit
`73c4aa1363a9bb2edfbe767740fcee0736428d81` (2026-10-25, Arnol Fokam and Tourad), built for the Apart AI Collusion Research
Sprint. That commit is this repo's starting point; everything after the first commit is new work (see `git log`).
`harness/tests/golden_legacy.json` is a snapshot of what the imported harness produced, so legacy games stay reproducible.

## Setup
Python 3.12+; `python -m venv .venv && .venv/bin/pip install -r requirements.txt` (`requirements-analysis.txt` for the
analysis). `export OPENROUTER_API_KEY=...`; Modal needs a secret named `openrouter` holding the key.
`python -m pytest harness/tests -q` runs every check without calling a model.
