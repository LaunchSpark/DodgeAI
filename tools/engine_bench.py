"""Measure the native engine's training-shaped throughput.

Steps 64 lanes on hard with patterns and powerups on, one frame per step,
reading the VelocityFlow observation every step as training does, with seeded
random actions. Prints one line, `frames/s: <float>`.

    python -m tools.engine_bench
"""

import time

import numpy as np

from dodge.env import make_engine

LANES = 64
WARMUP = 200
STEPS = 3_000


def main():
    engine = make_engine(step_frames=1, difficulty=3, patterns=True, powerups=True)
    rng = np.random.default_rng(0)
    engine.reset_ml_batch(np.arange(LANES, dtype=np.uint32))
    filler = 1_000_000
    frames = 0
    started = None
    for step in range(WARMUP + STEPS):
        if step == WARMUP:
            frames, started = 0, time.perf_counter()
        engine.observe_velocity()
        stepped = engine.step_ml_positions_batch(
            rng.integers(0, 9, LANES, dtype=np.uint8))
        frames += int(np.sum(stepped["frames_advanced"]))
        dead = np.flatnonzero(stepped["done"]).astype(np.uint32)
        if len(dead):
            engine.reset_ml_lanes(dead, np.arange(filler, filler + len(dead), dtype=np.uint32))
            filler += len(dead)
    print(f"frames/s: {frames / (time.perf_counter() - started):.0f}")


if __name__ == "__main__":
    main()
