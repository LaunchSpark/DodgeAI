"""Dodge-specific training adapters built on the generic PICO-8 runtime."""

from .observe import (
    CELL_COUNT,
    FRAME_OBSERVATION_SIZE,
    FRAME_STACK_SIZE,
    GRID_SIZE,
    OBSERVATION_SIZE,
    observe,
    observe_frame,
)

__all__ = [
    "CELL_COUNT",
    "FRAME_OBSERVATION_SIZE",
    "FRAME_STACK_SIZE",
    "GRID_SIZE",
    "OBSERVATION_SIZE",
    "observe",
    "observe_frame",
]
