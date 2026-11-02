"""What the critic may see that the actor may not.

The native engine appends hidden game state to the observation on request.
It is absent from an engine built before that existed, so the names here
fall back to an empty block rather than failing the import.
"""

try:
    import dodge_native
except ImportError:
    dodge_native = None

from .observe import OBSERVATION_SIZE

CRITIC_FEATURE_VERSION = getattr(dodge_native, "CRITIC_FEATURE_VERSION", 0)
CRITIC_FEATURE_COUNT = getattr(dodge_native, "CRITIC_FEATURE_COUNT", 0)
CRITIC_GROUPS = dict(getattr(dodge_native, "CRITIC_GROUPS", {}))
WIDE_OBSERVATION_SIZE = OBSERVATION_SIZE + CRITIC_FEATURE_COUNT
CRITIC_ARCHITECTURES = ("velocity-flow-v3",)


def critic_layout():
    """The layout as a checkpoint records it: plain, picklable, comparable."""
    return {"version": CRITIC_FEATURE_VERSION, "count": CRITIC_FEATURE_COUNT,
            "groups": {name: list(span) for name, span in CRITIC_GROUPS.items()}}
