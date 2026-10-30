"""Where game seeds come from, and which seeds training may never use.

Training used to draw `rng.integers(0, 2**31 - 1)`, which can land on an
evaluation seed: the paper's cohort is a contiguous block from 1,000,000, and
nothing kept training off it. Here a training game's seed is a hash of
(stream, run seed, env slot, that slot's episode index), so:

* it is a pure function of where the game sits in the run, so a run resumed
  from a checkpoint with the same per-slot episode indices plays the same
  games as one that never stopped;
* it never falls inside a registered reserved range -- a seed that hashes
  into one is rehashed with the next attempt number, so registering a new
  range later only moves the few seeds that fell inside it.

Evaluation still names its seeds explicitly (`ExplicitSeeds`, or a seed list
passed straight to the evaluator), reserved or not: those are the point.
"""

import hashlib
import struct

import numpy as np

U32 = 2**32
_MASK64 = 2**64 - 1
MAX_ATTEMPTS = 1_000

# The paper's evaluation (docs/paper/eval): every condition played seeds
# 1,000,000 onwards, at most 512 of them (docs/paper/eval/runs/*.jsonl). The
# block reserved here runs to 1,010,000, so a later confirmation cohort taken
# from the same block -- the paper notes it had none -- is protected too.
PAPER_EVAL_START = 1_000_000
PAPER_EVAL_USED = 512
PAPER_EVAL_STOP = 1_010_000

# The retention gate's fixed development cohort: the first DEV_COHORT_SIZE
# seeds of its block. The block is wider than the cohort so a larger cohort
# is still held out of training.
DEV_COHORT_START = 2_000_000
DEV_COHORT_SIZE = 256
DEV_COHORT_STOP = 2_010_000

# name -> (start, stop), half-open.
_RESERVED = {
    "paper-eval": (PAPER_EVAL_START, PAPER_EVAL_STOP),
    "dev": (DEV_COHORT_START, DEV_COHORT_STOP),
}


def reserved_ranges():
    """A copy of the registry: name -> (start, stop), half-open."""
    return dict(_RESERVED)


def register_reserved(name, start, stop):
    """Hold [start, stop) out of every training stream built from now on."""
    start, stop = int(start), int(stop)
    if not 0 <= start < stop <= U32:
        raise ValueError(f"reserved range must satisfy 0 <= start < stop <= 2**32, "
                         f"got [{start}, {stop})")
    _RESERVED[str(name)] = (start, stop)


def is_reserved(seed, ranges=None):
    ranges = _RESERVED if ranges is None else ranges
    seed = int(seed)
    return any(start <= seed < stop for start, stop in ranges.values())


def dev_cohort(size=DEV_COHORT_SIZE):
    """The fixed seeds the retention gate scores every candidate on."""
    if not 1 <= size <= DEV_COHORT_STOP - DEV_COHORT_START:
        raise ValueError(f"dev cohort size must be between 1 and "
                         f"{DEV_COHORT_STOP - DEV_COHORT_START}")
    return list(range(DEV_COHORT_START, DEV_COHORT_START + int(size)))


def _covered(ranges):
    """How many u32 values the ranges cover, overlaps counted once."""
    total, reach = 0, 0
    for start, stop in sorted(ranges.values()):
        start = max(start, reach)
        if stop > start:
            total += stop - start
            reach = stop
    return total


class SeedStream:
    """Deterministic per-slot training seeds that skip reserved ranges.

    `next_episode_indices[slot]` is the index of the next game that slot will
    start; it is all the state there is, so saving and restoring it resumes
    the stream exactly.
    """

    kind = "stream"

    def __init__(self, run_seed, n_slots, stream="train", reserved=None):
        self.run_seed = int(run_seed) & _MASK64
        self.stream = str(stream)
        self.reserved = dict(_RESERVED if reserved is None else reserved)
        if _covered(self.reserved) >= U32:
            raise ValueError("reserved ranges cover every seed")
        self.next_episode_indices = np.zeros(int(n_slots), dtype=np.int64)
        self._prefix = self.stream.encode("utf-8") + b"\x00"

    def _hash(self, slot, index, attempt):
        payload = self._prefix + struct.pack("<QQQQ", self.run_seed, int(slot),
                                             int(index), int(attempt))
        digest = hashlib.blake2b(payload, digest_size=4).digest()
        return int.from_bytes(digest, "little")

    def seed_for(self, slot, index):
        """The seed of `slot`'s `index`-th game. Does not advance anything."""
        for attempt in range(MAX_ATTEMPTS):
            seed = self._hash(slot, index, attempt)
            if not is_reserved(seed, self.reserved):
                return seed
        raise RuntimeError(f"no unreserved seed for slot {slot} episode {index} "
                           f"in {MAX_ATTEMPTS} attempts")

    def draw(self, slots):
        """The next seed for each slot in `slots`, advancing those slots."""
        slots = np.asarray(slots, dtype=np.int64).reshape(-1)
        out = np.empty(len(slots), dtype=np.uint32)
        for position, slot in enumerate(slots):
            out[position] = self.seed_for(slot, self.next_episode_indices[slot])
            self.next_episode_indices[slot] += 1
        return out

    def state_dict(self):
        return {
            "kind": self.kind,
            "run_seed": self.run_seed,
            "stream": self.stream,
            "next_episode_indices": [int(i) for i in self.next_episode_indices],
        }

    def load_state_dict(self, state):
        if state.get("kind", self.kind) != self.kind:
            raise ValueError(f"cannot restore a {state.get('kind')} seed state "
                             f"into a {self.kind}")
        self.run_seed = int(state["run_seed"]) & _MASK64
        self.stream = str(state["stream"])
        self._prefix = self.stream.encode("utf-8") + b"\x00"
        indices = np.asarray(state["next_episode_indices"], dtype=np.int64)
        if len(indices) != len(self.next_episode_indices):
            raise ValueError(f"seed state has {len(indices)} slots, this stream "
                             f"has {len(self.next_episode_indices)}")
        self.next_episode_indices = indices.copy()


class ExplicitSeeds:
    """A fixed seed list, dealt in order to whichever slots ask next.

    Evaluation's mode: the list is the cohort, so reserved ranges do not apply,
    and running out is an error rather than a silent switch to other seeds.
    """

    kind = "explicit"

    def __init__(self, seeds, n_slots=1):
        self.seeds = np.asarray(list(seeds), dtype=np.uint32)
        self.position = 0
        self.next_episode_indices = np.zeros(int(n_slots), dtype=np.int64)

    def draw(self, slots):
        slots = np.asarray(slots, dtype=np.int64).reshape(-1)
        if self.position + len(slots) > len(self.seeds):
            raise RuntimeError(f"explicit seed list exhausted: {len(self.seeds)} "
                               f"seeds, {self.position + len(slots)} games asked for")
        out = self.seeds[self.position:self.position + len(slots)].copy()
        self.position += len(slots)
        np.add.at(self.next_episode_indices, slots, 1)
        return out

    def state_dict(self):
        return {
            "kind": self.kind,
            "position": int(self.position),
            "next_episode_indices": [int(i) for i in self.next_episode_indices],
        }

    def load_state_dict(self, state):
        if state.get("kind", self.kind) != self.kind:
            raise ValueError(f"cannot restore a {state.get('kind')} seed state "
                             f"into a {self.kind}")
        self.position = int(state["position"])
        self.next_episode_indices = np.asarray(state["next_episode_indices"],
                                               dtype=np.int64).copy()
