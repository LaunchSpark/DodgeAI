"""The game mechanics the dashboard's Game Config panel owns, kept on disk.

Unlike the reward weights, these do not reach the environments through the
file: applying a change rebuilds the training games from the session's env
kwargs, so the running processes are told directly. The file exists for the
other half of the problem -- a launch used to start from the command line's
defaults no matter what the last session had been tuned to, silently throwing
away a curriculum someone had dialled in. It is written on Apply and read at
startup, and is as hand-editable as rewards.json.
"""

import json
from pathlib import Path


CONFIG_PATH = Path("game-config.json")

# The panel's settings and their shipped values. Anything else in the
# session's env kwargs -- the frame cap, for one -- is not the panel's to
# store, so it is neither read nor written here.
DEFAULTS = {
    "difficulty": 1,
    "patterns": False,
    "powerups": True,
    "normal_enemies": True,
    "kamikaze_enemies": True,
    "spawn_rate_percent": 100,
    "action_repeat": 4,
    "lives": 0,
}

# Ranges the session's configure_game enforces. Held here too so a hand edit
# is clamped on the way in rather than raising once training tries to start.
BOUNDS = {
    "difficulty": (1, 3),
    "spawn_rate_percent": (50, 150),
    "action_repeat": (1, 5),
    "lives": (0, 5),
}


def _clean(key, value):
    """Coerce one hand-written value, or None if it cannot be used."""
    if key in BOUNDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        minimum, maximum = BOUNDS[key]
        return min(maximum, max(minimum, int(value)))
    # Everything else is a toggle. JSON booleans are the normal spelling, but
    # a 0/1 from a hand edit means the obvious thing.
    if isinstance(value, (bool, int)):
        return bool(value)
    return None


def load(path=CONFIG_PATH):
    """Read the saved settings, falling back to defaults for anything missing.

    A malformed or absent file yields defaults rather than raising: the
    dashboard must not fail to launch because a hand edit left a trailing
    comma.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULTS)
    if not isinstance(raw, dict):
        return dict(DEFAULTS)
    config = dict(DEFAULTS)
    for key in DEFAULTS:
        if key not in raw:
            continue
        value = _clean(key, raw[key])
        if value is not None:
            config[key] = value
    return config


def save(config, path=CONFIG_PATH):
    """Write the panel's settings, ignoring any other key it was handed."""
    path = Path(path)
    stored = {key: _clean(key, config[key]) if key in config else None
              for key in DEFAULTS}
    stored = {key: DEFAULTS[key] if value is None else value
              for key, value in stored.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
