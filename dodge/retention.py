"""The retention gate: a checkpoint is only a candidate until it earns its place.

Every candidate is scored on the fixed dev cohort (dodge.seeds.dev_cohort,
held out of training) with argmax actions, by the no-replacement evaluator.
It is promoted to the retained best -- `retained/<name>-best.zip` with its state
sidecar and manifest, byte-for-byte copies of the candidate's -- only if its
dev mean beats the best so far. Training can restart from the retained best
like any other checkpoint, with model, Adam and every generator restored.

Beside the best: `<name>-best.retention.json` (why it is the best: its dev
scores and which candidate it was) and `<name>-best.history.jsonl` (every
candidate's dev scores, promoted or not).
"""

from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import shutil

from .evaluate import engine_settings, evaluate_model
from .lineage import checkpoint_path, sha256_file, sidecar_paths
from .seeds import dev_cohort

# The paper's evaluation cap: 18,000 frames is five minutes at 60 Hz.
DEV_FRAME_CAP = 18_000


# The best and the candidate are the gate's working files, not models anyone
# picks, so they live a folder down: the dashboard lists every checkpoint at
# the top of the models folder, and had been offering them beside the model
# itself.
RETAINED_DIR = "retained"


def best_path_for(models_dir, name):
    return Path(models_dir) / RETAINED_DIR / f"{name}-best.zip"


def candidate_path_for(models_dir, name):
    # No ".zip": SB3 appends it when saving.
    return Path(models_dir) / RETAINED_DIR / f"{name}-candidate"


def move_legacy_retention(models_dir, name):
    """Move `<name>-best.*` and `<name>-candidate.*` into RETAINED_DIR.

    They used to sit at the top of the models folder. Moved rather than left,
    since a gate that finds no record promotes whatever it scores first, and
    a file already in the folder is never overwritten.
    """
    models_dir = Path(models_dir)
    target = models_dir / RETAINED_DIR
    for suffix in ("best", "candidate"):
        for source in models_dir.glob(f"{glob.escape(name)}-{suffix}.*"):
            moved = target / source.name
            if source.is_file() and not moved.exists():
                target.mkdir(parents=True, exist_ok=True)
                os.replace(source, moved)


def _stem(path):
    path = checkpoint_path(path)
    return path.with_suffix("") if path.suffix == ".zip" else path


def _copy_atomically(source, target):
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


class RetentionGate:
    def __init__(self, best_path, *, game_settings, seeds=None,
                 frame_cap=DEV_FRAME_CAP, evaluate=evaluate_model):
        self.best_path = checkpoint_path(best_path)
        self.record_path = Path(f"{_stem(self.best_path)}.retention.json")
        self.history_path = Path(f"{_stem(self.best_path)}.history.jsonl")
        self.game = engine_settings(game_settings)
        self.seeds = list(dev_cohort() if seeds is None else seeds)
        self.frame_cap = frame_cap
        self.evaluate = evaluate

    @property
    def record(self):
        if not self.record_path.exists():
            return None
        return json.loads(self.record_path.read_text(encoding="utf-8"))

    @property
    def best_mean(self):
        record = self.record
        return None if record is None else record["dev"]["mean"]

    def consider(self, model, candidate, *, update=None):
        """Score `model` (saved at `candidate`); promote it if it improves."""
        candidate = checkpoint_path(candidate)
        result = self.evaluate(model, self.seeds, deterministic=True,
                               frame_cap=self.frame_cap, **self.game)
        dev = result.summary()
        previous = self.best_mean
        promoted = previous is None or dev["mean"] > previous
        entry = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": str(candidate),
            "candidate_sha256": sha256_file(candidate),
            "update": update,
            "dev": dev,
            "previous_best_mean": previous,
            "promoted": promoted,
            "seeds": [self.seeds[0], self.seeds[-1], len(self.seeds)],
            "frame_cap": self.frame_cap,
        }
        if promoted:
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            for source, target in zip((candidate, *sidecar_paths(candidate)),
                                      (self.best_path, *sidecar_paths(self.best_path))):
                if source.exists():
                    _copy_atomically(source, target)
            record = {**entry, "games": result.to_dict()["games"]}
            temporary = self.record_path.with_name(f".{self.record_path.name}.tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.record_path)
        with open(self.history_path, "a", encoding="utf-8") as history:
            history.write(json.dumps(entry) + "\n")
        return {"promoted": promoted, "dev": dev, "best_mean": self.best_mean}
