#!/usr/bin/env python3
"""Write a freshly initialised policy checkpoint.

    python tools/new_policy.py field-cnn

The dashboard's Models panel lists the checkpoints on disk, and loading one
switches the running session to that architecture. So an architecture with no
saved checkpoint cannot be reached from the dashboard at all -- only by
relaunching with --architecture. This writes an untrained starting point so a
new architecture is selectable straight away.
"""

import argparse
from pathlib import Path
import sys

# Run directly from anywhere, like the other tools here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from dodge.env import DodgeEnv
from dodge.policies import ARCHITECTURES, architecture_name, policy_class, policy_kwargs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture", choices=ARCHITECTURES)
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--name", default=None,
                        help="checkpoint name (defaults to the architecture)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    # A throwaway environment: only its observation and action spaces are
    # recorded, and the session rebinds the real ones when it loads this.
    env = DodgeEnv(max_episode_frames=20)
    model = PPO(
        policy_class(args.architecture), env,
        n_steps=64, batch_size=64, seed=args.seed, device="cpu",
        policy_kwargs=policy_kwargs(args.architecture),
    )
    env.close()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    path = args.models_dir / (args.name or args.architecture)
    model.save(path)
    parameters = sum(p.numel() for p in model.policy.parameters())
    print(f"Wrote {path.with_suffix('.zip')} "
          f"({architecture_name(model)}, {parameters:,} parameters, untrained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
