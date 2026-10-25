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

## The experiment
- **Agents:** `nousresearch/hermes-4-405b`, `cognitivecomputations/dolphin-mistral-24b-venice-edition`, `anthropic/claude-opus-5.5`.
- **Tasks:** `team_leaderboard_v2`, `team_database`, `team_auction`, `team_coding` and the new `team_doc_qa`
  (`harness/doc_qa.py`). Each agent answers 10 items: 5 public, plus the team's 5 special items.
- **Honest-path difficulty** for the special items (`difficulty`): written in a public file (D0), one mapping step (D1),
  a join of several sources plus a rule (D2), or no honest path (D3).
- **The leak** is the previous version of the key (v1). The scorer accepts it, and it appears nowhere else, so v1
  answers prove leak use at every level.
- **Metrics:** leak used, leak shared, collusion happened, collusion detected (a passive security officer, full view and
  messages-only), plus the control teams' honest solve rate as the manipulation check. See `experiments/PREREG.md`.

## Running
```bash
python -m pytest harness/tests -q                                   # every check, no model calls
python -m harness.run_modal --agent hermes                           # dry run: games, settings, estimated cost
python -m harness.run_team_local team_doc_qa 200 2 collude           # one scripted game locally, with its transcript
modal run harness/run_modal.py --mode difficulty --agent dolphin --tasks team_coding --levels 0,3 --seeds 1 --tag smoke --budget 2
modal run harness/run_modal.py --mode difficulty --agent hermes --tag main --budget 40 [--resume]
python -m harness.analysis_difficulty results/difficulty-*.jsonl     # tables, tests, figure
python analysis/verify.py results/difficulty-*.jsonl --replay        # numbers.json behind every reported number
```

## Transcripts
Every game writes a complete transcript to `transcripts/<run>/<game>.json.gz` (`harness/transcript.py`). It holds:
- each agent's full brief and starting files;
- every model call: the exact observation, the full reply, tokens and latency;
- every command, with its full sandbox output;
- every message, flagged when it carries a leak (v1) or honestly derived answers (v2);
- crossings and scores;
- the full prompts and replies of the officer, the observer and the interviews.

Each results line stores the transcript's path and sha256. `python -m harness.transcript <file>` replays a game and checks
that it reproduces the recorded score. `python -m harness.transcript_view --index transcripts/<run>/` writes one HTML page
per game plus an index.
