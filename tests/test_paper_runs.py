"""The published evaluation runs counted each seed exactly once.

docs/paper/eval/evalharness.py keeps a fixed list of N seeds, refills a
finished lane only with the next unplayed seed from that list, and stops when
every one of them has finished -- so short games are not over-counted, the
way they would be by stopping after N episodes of an auto-resetting VecEnv.
These checks pin that down on the recorded data the paper's tables came from.
"""

import json
from pathlib import Path

import pytest

from dodge import seeds

RUNS = Path(__file__).resolve().parents[1] / "docs" / "paper" / "eval" / "runs"

# docs/paper/eval/progress.py's plan: run -> games.
PLANNED = {
    "latest_k1_argmax": 512,
    "latest_k1_sampled": 512,
    "latest_k4_argmax": 512,
    "latest_k4_sampled": 512,
    "c100_k1_argmax": 256,
    "random_k1": 256,
    "gradient_k1": 128,
}


@pytest.mark.parametrize("run,games", sorted(PLANNED.items()))
def test_every_seed_of_the_cohort_appears_exactly_once(run, games):
    path = RUNS / f"{run}.jsonl"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    played = [json.loads(line)["seed"] for line in path.read_text().splitlines()
              if line.strip()]
    assert sorted(played) == list(range(1_000_000, 1_000_000 + games))


def test_the_paper_cohort_is_held_out_of_training():
    for seed in range(1_000_000, 1_000_000 + max(PLANNED.values())):
        assert seeds.is_reserved(seed)
