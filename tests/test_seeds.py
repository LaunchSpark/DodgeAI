"""Seed streams: deterministic game seeds that never land on held-out cohorts."""

import numpy as np
import pytest

from dodge import seeds
from dodge.seeds import ExplicitSeeds, SeedStream


def test_a_game_seed_is_a_pure_function_of_run_slot_and_episode():
    a = SeedStream(run_seed=7, n_slots=4)
    b = SeedStream(run_seed=7, n_slots=4)
    assert a.seed_for(2, 5) == b.seed_for(2, 5)
    assert a.seed_for(2, 5) != a.seed_for(3, 5)
    assert a.seed_for(2, 5) != a.seed_for(2, 6)
    assert SeedStream(run_seed=8, n_slots=4).seed_for(2, 5) != a.seed_for(2, 5)
    assert SeedStream(run_seed=7, n_slots=4, stream="other").seed_for(2, 5) \
        != a.seed_for(2, 5)


def test_seeds_are_u32():
    stream = SeedStream(run_seed=1, n_slots=64)
    drawn = np.concatenate([stream.draw(np.arange(64)) for _ in range(20)])
    assert drawn.dtype == np.uint32
    # 1,280 draws spread over 32 bits: some land above 2**31, which the old
    # 31-bit draw never produced.
    assert drawn.max() >= 2**31
    assert len(np.unique(drawn)) == len(drawn)


def test_drawing_advances_only_the_drawn_slots():
    stream = SeedStream(run_seed=3, n_slots=3)
    first = stream.draw(np.array([0, 2]))
    assert list(stream.next_episode_indices) == [1, 0, 1]
    second = stream.draw(np.array([2]))
    assert list(stream.next_episode_indices) == [1, 0, 2]
    assert first[1] == stream.seed_for(2, 0)
    assert second[0] == stream.seed_for(2, 1)


def test_state_round_trips():
    stream = SeedStream(run_seed=11, n_slots=2)
    stream.draw(np.array([0, 1, 1]))
    restored = SeedStream(run_seed=0, n_slots=2)
    restored.load_state_dict(stream.state_dict())
    assert restored.run_seed == 11
    assert np.array_equal(restored.draw(np.array([0, 1])), stream.draw(np.array([0, 1])))


def test_training_seeds_skip_reserved_ranges():
    # Reserve the bottom half of the space: every draw must still come out
    # above the reserved block, and deterministically so.
    reserved = {"bottom-half": (0, 2**31)}
    stream = SeedStream(run_seed=5, n_slots=8, reserved=reserved)
    drawn = np.concatenate([stream.draw(np.arange(8)) for _ in range(4)])
    assert drawn.min() >= 2**31
    again = SeedStream(run_seed=5, n_slots=8, reserved=reserved)
    assert np.array_equal(np.concatenate([again.draw(np.arange(8)) for _ in range(4)]),
                          drawn)


def test_the_paper_cohort_and_the_dev_cohort_are_reserved():
    ranges = seeds.reserved_ranges()
    # The paper's evaluation: 512 seeds from 1,000,000 (docs/paper/eval/runs).
    start, stop = ranges["paper-eval"]
    assert start <= 1_000_000 and stop >= 1_000_512
    cohort = seeds.dev_cohort()
    assert len(cohort) == seeds.DEV_COHORT_SIZE == 256
    assert all(seeds.is_reserved(seed) for seed in cohort)
    assert all(seeds.is_reserved(seed) for seed in range(1_000_000, 1_000_512))
    assert not seeds.is_reserved(42)
    # The cohorts are disjoint from each other.
    assert not set(cohort) & set(range(1_000_000, 1_000_512))


def test_registering_a_range_protects_it_from_new_streams(monkeypatch):
    monkeypatch.setattr(seeds, "_RESERVED", dict(seeds._RESERVED))
    seeds.register_reserved("most", 0, 3 * 2**30)
    stream = SeedStream(run_seed=1, n_slots=4)
    assert stream.draw(np.arange(4)).min() >= 3 * 2**30
    with pytest.raises(ValueError):
        seeds.register_reserved("backwards", 10, 5)


def test_a_reserved_range_cannot_cover_every_seed():
    with pytest.raises(ValueError):
        SeedStream(run_seed=1, n_slots=1, reserved={"all": (0, 2**32)})


def test_explicit_seeds_are_dealt_in_order_and_may_be_reserved():
    dealer = ExplicitSeeds([1_000_000, 1_000_001, 1_000_002], n_slots=2)
    assert list(dealer.draw(np.array([0, 1]))) == [1_000_000, 1_000_001]
    assert list(dealer.draw(np.array([1]))) == [1_000_002]
    assert list(dealer.next_episode_indices) == [1, 2]
    with pytest.raises(RuntimeError):
        dealer.draw(np.array([0]))


def test_explicit_state_round_trips():
    dealer = ExplicitSeeds([5, 6, 7], n_slots=2)
    dealer.draw(np.array([0]))
    copy = ExplicitSeeds([5, 6, 7], n_slots=2)
    copy.load_state_dict(dealer.state_dict())
    assert list(copy.draw(np.array([1, 0]))) == [6, 7]
