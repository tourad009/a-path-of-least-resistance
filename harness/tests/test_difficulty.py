"""Honest-path difficulty: legacy games are unchanged, every level's honest path works from public files only, and the
previous-version key (v1) proves leak use at every level."""
import json

import pytest

from harness.tests import golden_snapshot


def test_legacy_games_match_the_golden_snapshot():
    """Games built without `difficulty` give the same briefs, files, officer prompts and scripted scores as the imported
    harness (collusion-leaderboard@73c4aa1)."""
    want = json.loads(golden_snapshot.PATH.read_text())
    got = json.loads(json.dumps(golden_snapshot.snapshot()))
    assert set(got) == set(want)
    for key in want:
        for field in want[key]:
            assert got[key][field] == want[key][field], f"{key}: {field} changed"
