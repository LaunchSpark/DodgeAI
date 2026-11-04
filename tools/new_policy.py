#!/usr/bin/env python3
"""Write a freshly initialised policy checkpoint.

    python tools/new_policy.py velocity-flow-v3
    python tools/new_policy.py velocity-flow-v3 --name v3-velocity-only \\
        --no-graduated-holds --no-centred-logits

The dashboard's Training panel does the same from "Start a new model"; see
dodge.new_policy. This is the command-line route to it.
"""

import argparse
from pathlib import Path
import sys

# Run directly from anywhere, like the other tools here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dodge.new_policy import ACTOR_FLAG_TEXT, write_new_policy
from dodge.policies import ARCHITECTURES, V3_ACTOR_FLAGS, architecture_name
from dodge.training import DashboardPPO


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("architecture", choices=ARCHITECTURES)
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--name", default=None,
                        help="checkpoint name (defaults to the architecture)")
    parser.add_argument("--seed", type=int, default=0)
    for flag in V3_ACTOR_FLAGS:
        parser.add_argument(f"--no-{flag.replace('_', '-')}", dest=flag,
                            action="store_false", default=None,
                            help=f"velocity-flow-v3 only: {ACTOR_FLAG_TEXT[flag][1].lower()}, off")
    args = parser.parse_args(argv)

    flags = {flag: getattr(args, flag) for flag in V3_ACTOR_FLAGS
             if getattr(args, flag) is not None}
    try:
        path = write_new_policy(args.architecture, args.models_dir, name=args.name,
                                seed=args.seed, **flags)
    except ValueError as error:
        parser.error(str(error))
    model = DashboardPPO.load(path, device="cpu")
    parameters = sum(p.numel() for p in model.policy.parameters())
    print(f"Wrote {path} ({architecture_name(model)}, {parameters:,} parameters, untrained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
