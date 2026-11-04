"""Tunable training schedule, shared by the dashboard and a text editor.

Like rewards.json, a file is the source of truth: the dashboard writes it and
the training session re-reads it at every update boundary, so a change takes
effect from the next PPO update without restarting the run.

What it holds is the plan for continuing a trained model rather than starting
one: a learning rate that steps down in stages when the dev cohort stops
improving, a discount, and how often to score that cohort. Each scoring runs
through the retention gate, so the best checkpoint so far is always on disk as
`<name>-best.zip` whatever the latest weights do.

A missing file means "the architecture's table": nothing is applied, so a run
that never opens the Training panel trains exactly as it did before.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .rewards import SliderSpec


CONFIG_PATH = Path("training.json")

# The stages the learning rate steps down through, highest first. A rate set by
# hand between two stages steps to the next one below it.
LR_STAGES = (3e-4, 1e-4, 3e-5)

# Dev scorings averaged before the plateau rule looks at them. One scoring of
# the 256-seed cohort carries about +-320 frames of sampling noise, and
# successive checkpoints correlate at |r| <= 0.13, so a raw best is mostly the
# luckiest bounce so far: it sets a bar the next few scorings cannot reach, and
# the rate steps down on a run that is still climbing. Three scorings cut a
# bounce to a third of its height.
ROLLING_SCORINGS = 3


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    # v2's table value, so a file written without a discount does not impose
    # a shorter one over the table. 0.99 was the old default here, and at one
    # frame a decision it is a horizon of about 100 frames, shorter than the
    # 125-frame warning a pattern gives before it turns lethal.
    gamma: float = 0.997
    # PPO updates between dev-cohort scorings; 0 turns scoring off, and with it
    # the automatic learning-rate steps.
    retain_every: int = 60
    # Scorings in a row without a new best rolling mean (ROLLING_SCORINGS of
    # them) before the learning rate steps down a stage; 0 never steps it.
    patience: int = 3

    def clamped(self):
        values = asdict(self)
        for spec in SLIDERS:
            value = min(spec.maximum, max(spec.minimum, float(values[spec.name])))
            values[spec.name] = (int(round(value)) if spec.name in INTEGER_FIELDS
                                 else value)
        return TrainingConfig(**values)


INTEGER_FIELDS = ("retain_every", "patience")

SLIDERS = (
    SliderSpec("learning_rate", "Learning rate", 1e-5, 1e-3, 1e-5,
               "Steps down 3e-4 → 1e-4 → 3e-5 when the dev cohort plateaus"),
    SliderSpec("gamma", "Discount (γ)", 0.98, 0.999, 0.001,
               "0.99 looks ~100 frames ahead, 0.997 ~330"),
    SliderSpec("retain_every", "Score dev cohort every", 0, 240, 10,
               "PPO updates between scorings of the held-out seeds; 0 is off"),
    SliderSpec("patience", "Plateau patience", 0, 10, 1,
               "Scorings without a new best 3-scoring mean before the rate "
               "steps down; 0 never"),
)


def next_stage(learning_rate):
    """The stage below `learning_rate`, or None if it is already at the last."""
    lower = [stage for stage in LR_STAGES if stage < learning_rate * (1 - 1e-6)]
    return lower[0] if lower else None


def exists(path=CONFIG_PATH):
    return Path(path).exists()


def load(path=CONFIG_PATH):
    """Read the config, falling back to defaults for anything missing.

    A malformed file yields defaults rather than raising, as rewards.load does:
    training must not die because a hand edit left a trailing comma.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return TrainingConfig()
    if not isinstance(raw, dict):
        return TrainingConfig()
    known = {field.name for field in fields(TrainingConfig)}
    values = {key: value for key, value in raw.items()
              if key in known and isinstance(value, (int, float))}
    return TrainingConfig(**values).clamped()


def save(config, path=CONFIG_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(config), indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return path
