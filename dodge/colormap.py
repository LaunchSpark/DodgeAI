"""Viridis, as a lookup table.

Copied in rather than imported, because matplotlib is a large dependency to
add for 256 RGB triples and the training install should stay small. The
control points below are matplotlib's own viridis sampled at every tenth,
interpolated to 256 entries at import.

Viridis suits this job for the reason it was designed: it is perceptually
uniform and monotonic in lightness, so a danger gradient reads as a gradient
rather than as banding, and it stays legible for red-green colour blindness.
It also happens not to collide with anything Dodge draws -- the game is white,
black and one accent colour, so a blue-to-yellow field sits behind the action
without competing with it.
"""

import numpy as np

_CONTROL_POINTS = (
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
    (180, 222, 44), (253, 231, 37),
)
LEVELS = 256


def _build(points=_CONTROL_POINTS, levels=LEVELS):
    source = np.asarray(points, dtype=np.float64)
    at = np.linspace(0.0, 1.0, len(source))
    want = np.linspace(0.0, 1.0, levels)
    channels = [np.interp(want, at, source[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=1).round().astype(np.uint8)


VIRIDIS = _build()


def colorize(values, low, high):
    """Map an array of scalars to RGB, clamped to the [low, high] range."""
    span = max(float(high) - float(low), 1e-6)
    normalised = (np.asarray(values, dtype=np.float32) - float(low)) / span
    index = np.clip(normalised * (LEVELS - 1), 0, LEVELS - 1).astype(np.int32)
    return VIRIDIS[index]
