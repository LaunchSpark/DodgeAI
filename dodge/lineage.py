"""Provenance manifests: what produced a checkpoint, and what it came from.

Each SB3 checkpoint `X.zip` gets `X.manifest.json` beside it, recording:

* sha256 of the model zip and of its state sidecar (`X.state.pt`);
* the full configuration actually in force -- the model's hyperparameters,
  the env's settings and the reward configuration's contents;
* runtime versions, and the sha256 of the installed `dodge_native` binary;
* the git commit of this code, and whether the tree was dirty;
* the parent checkpoint's sha256 and its manifest's sha256, so a lineage
  chains back through every checkpoint it passed through;
* the run's counters and its wall-clock seconds, the parent's included.

The paper's limitations section notes that the settings in force during
training "are not stored with the checkpoints". This is that record.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

STATE_SUFFIX = ".state.pt"
MANIFEST_SUFFIX = ".manifest.json"
MANIFEST_FORMAT = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def checkpoint_path(path):
    """Where SB3's save() puts `path`: it appends .zip only to a bare name."""
    path = Path(path)
    return path if path.suffix else path.with_suffix(".zip")


def sidecar_paths(checkpoint):
    """(state sidecar, manifest) paths for a checkpoint zip."""
    checkpoint = checkpoint_path(checkpoint)
    stem = checkpoint.with_suffix("") if checkpoint.suffix == ".zip" else checkpoint
    return Path(f"{stem}{STATE_SUFFIX}"), Path(f"{stem}{MANIFEST_SUFFIX}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions():
    import gymnasium
    import numpy
    import stable_baselines3
    import torch
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "platform": platform.platform(),
    }


def native_binary():
    """The installed dodge_native extension and its sha256, or None."""
    try:
        from dodge_native import dodge_native as extension
    except ImportError:
        return None
    path = Path(extension.__file__).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def git_state(root=PROJECT_ROOT):
    """The commit this code is at and whether tracked or new files differ."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, timeout=30, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def jsonable(value):
    """A JSON-safe rendering: classes by qualified name, the rest by repr."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy
        if isinstance(value, numpy.generic):
            return value.item()
    except ImportError:  # pragma: no cover
        pass
    return repr(value)


def _scheduled(value):
    """A hyperparameter that SB3 may hold as a schedule, at its start."""
    return float(value(1.0)) if callable(value) else value


def model_config(model):
    """The hyperparameters a model is actually running with."""
    return jsonable({
        "algorithm": type(model).__name__,
        "policy_class": type(model.policy),
        "learning_rate": _scheduled(model.learning_rate),
        "n_steps": getattr(model, "n_steps", None),
        "batch_size": getattr(model, "batch_size", None),
        "n_epochs": getattr(model, "n_epochs", None),
        "gamma": model.gamma,
        "gae_lambda": getattr(model, "gae_lambda", None),
        "clip_range": _scheduled(getattr(model, "clip_range", None)),
        "clip_range_vf": _scheduled(getattr(model, "clip_range_vf", None)),
        "normalize_advantage": getattr(model, "normalize_advantage", None),
        "ent_coef": getattr(model, "ent_coef", None),
        "vf_coef": getattr(model, "vf_coef", None),
        "max_grad_norm": getattr(model, "max_grad_norm", None),
        "target_kl": getattr(model, "target_kl", None),
        "use_sde": getattr(model, "use_sde", None),
        "seed": model.seed,
        "device": str(model.device),
        "n_envs": model.n_envs,
        "policy_kwargs": model.policy_kwargs,
        "optimizer": type(model.policy.optimizer),
    })


def read_manifest(checkpoint):
    path = sidecar_paths(checkpoint)[1]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parent_reference(checkpoint):
    """How a child manifest names its parent: by content, not only by path."""
    checkpoint = checkpoint_path(checkpoint)
    manifest = sidecar_paths(checkpoint)[1]
    return {
        "path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
    }


def write_manifest(checkpoint, *, config, counters, parent, wall_clock_seconds,
                   extra=None):
    """Write `checkpoint`'s manifest beside it and return the manifest's path."""
    checkpoint = checkpoint_path(checkpoint)
    state_path, manifest_path = sidecar_paths(checkpoint)
    manifest = {
        "format": MANIFEST_FORMAT,
        "checkpoint": checkpoint.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha256_file(checkpoint),
        "state_sha256": sha256_file(state_path) if state_path.exists() else None,
        "config": jsonable(config),
        "versions": runtime_versions(),
        "dodge_native": native_binary(),
        "git": git_state(),
        "parent": parent,
        "counters": jsonable(counters),
        "wall_clock_seconds": float(wall_clock_seconds),
    }
    if extra:
        manifest.update(jsonable(extra))
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path
