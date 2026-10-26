# a-path-of-least-resistance

**A Path of Least Resistance: How Language Model Agents Collude When the Honest Route Disappears.**
Research conducted at the Apart AI Collusion Research Sprint, October 2026 (Tourad and Arnol Fokam).

A multi-team testbed where one member of some teams (the *holder*) finds a leaked answer file. The collusive path (the
leak) stays fixed, while the **honest path** to the same points is made easy (D0), harder (D1, D2) or impossible (D3). We
measure whether the leak is used, shared and taken up, and whether a passive security officer detects it.

## Status
- **Main experiment: complete.** 600 runs: 6 models × 5 tasks × 4 levels × 5 seeds. Results are in `results/difficulty-*-main.jsonl`,
  with one transcript per run in `transcripts/`.
- **Ablations: designed and partly run.** The design and predictions were fixed before any run
  (`experiments/ABLATIONS.md`). Partial results so far (`results/ablations-*-ab1.jsonl`):
  - leak handed over (`handover`), 120 of 150 runs;
  - messaging off (`nomessaging`), 70 of 150;
  - password leak and its answer-leak baseline, 10 runs each.

  The other arms have not been run.
- **Officer re-checks:** the 600 main runs read again with the original prompt and by a second-vendor officer
  (`results/*-main-officer-*.jsonl`).
- **Paper:** `paper/` (LaTeX). Every number, table and figure is generated from the results files.

## Repository layout
| Path | What it holds |
|---|---|
| `harness/` | the game, the five tasks, the honest-path levels, the officer, transcripts, the Modal runner, and the analysis |
| `harness/tests/` | every check, with no model calls: scorer validation, canary invariants, transcript replay, runner |
| `experiments/` | `PREREG.md` (main experiment) and `ABLATIONS.md` (ablations, frozen before running) |
| `results/` | one JSON line per run |
| `transcripts/` | a full transcript per run (`.json.gz`) |
| `analysis/` | `verify.py`, which recomputes every reported number into `numbers.json` |
| `paper/` | the paper: `main.tex`, `gen_paper.py` (numbers, prompts and tables from the harness), and the figures |

## The experiment
- **Agents (six models):** `nousresearch/hermes-4-405b`, `meta-llama/llama-4-maverick`, `deepseek/deepseek-v3.2`,
  `cognitivecomputations/dolphin-mistral-24b-venice-edition`, `anthropic/claude-haiku-5.5`, `anthropic/claude-opus-5.5`,
  all called through OpenRouter.
- **Tasks:** `team_leaderboard_v2`, `team_database`, `team_auction`, `team_coding` and `team_doc_qa`. Each agent answers 10
  items: 5 public, plus the team's 5 special items.
- **Honest-path difficulty** for the special items (`difficulty`):
  - D0: the answer is written in a public file;
  - D1: one mapping step;
  - D2: a join of several sources plus a rule;
  - D3: no honest path.
- **The leak** is the previous version of the key (v1). The scorer accepts it, and it appears nowhere else, so v1 answers
  prove leak use at every level. A password variant (`team_database_password`) is used in the ablations; its proof of use
  is the login.
- **Metrics:**
  - leak used;
  - leak shared;
  - collusion happened;
  - collusion detected, by a security officer that reads each finished run and never acts during it, in a full view
    and a messages-only view;
  - plus the control teams' honest solve rate as the manipulation check.

## Setup
- Python 3.12 or later:
  ```
  python -m venv .venv
  .venv/bin/pip install -r requirements.txt -r requirements-analysis.txt
  ```
- Model calls go through OpenRouter (`export OPENROUTER_API_KEY=...`). Runs execute on [Modal](https://modal.com), which
  needs a secret named `openrouter` holding the key.

## Running
```bash
python -m pytest harness/tests -q                                    # every check, no model calls
python -m harness.run_modal --agent hermes                            # dry run: games, settings, estimated cost
python -m harness.run_team_local team_doc_qa 200 2 collude            # one scripted game locally, with its transcript

modal run harness/run_modal.py --mode key-check                       # OpenRouter key limit and account balance
modal run harness/run_modal.py --mode difficulty --agent hermes --tag main --budget 40 [--resume]
MAX_GAMES=300 modal run harness/run_modal.py --mode ablations --tag ab1 --budget 400 [--arms handover,warning] [--resume]
modal run harness/run_modal.py --mode rereview --src 'results/difficulty-*-main.jsonl' \
    --officer-model openai/gpt-4.1-mini --tag gpt41mini                # the officer reads finished runs again

python -m harness.analysis_difficulty results/difficulty-*-main.jsonl  # tables, tests, figure
python analysis/verify.py results/difficulty-*-main.jsonl --replay     # numbers.json behind every reported number
make -C paper                                                          # regenerate the paper's numbers and build the PDF (tectonic)
```
`--agent` takes OpenRouter ids or short names (`hermes`, `maverick`, `deepseek`, `dolphin`, `haiku`, `opus`), several at
once with commas. Every launch needs `--budget`; `--resume` skips runs already in the results file.

## Transcripts
Every run writes a complete transcript to `transcripts/<run>/<game>.json.gz` (`harness/transcript.py`). It holds:
- each agent's full brief and starting files;
- every model call: the exact observation sent, the full reply, tokens and latency;
- every command, with its full sandbox output;
- every message, flagged when it carries a leak (v1) or honestly derived answers (v2);
- crossings and scores;
- the full prompts and replies of the officer, the observer and the interviews.

Each results line stores its transcript's path and sha256.
- `python -m harness.transcript <file>` replays a run and checks that it reproduces the recorded score.
- `python -m harness.transcript_view --index transcripts/<run>/` writes one HTML page per run, plus an index.

## Provenance
The harness was imported from [`ArnolFokam/collusion-leaderboard`](https://github.com/ArnolFokam/collusion-leaderboard),
branch `submission`, commit `73c4aa1` (25 October 2026, Arnol Fokam and Tourad), built during the same sprint. That commit
is this repository's starting point; everything after the first commit is new work (see `git log`).
`harness/tests/golden_legacy.json` records what the imported harness produced, so legacy games stay reproducible.
