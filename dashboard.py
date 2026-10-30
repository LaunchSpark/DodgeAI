#!/usr/bin/env python3
"""Train Dodge PPO bots from a browser dashboard.

    python dashboard.py                         # resume the saved game settings
    python dashboard.py --architecture velocity-flow-v2 --envs 32

Serves the dashboard at http://127.0.0.1:8765 and opens it. The page is built
from web/ with `pnpm install && pnpm run build`; during frontend development
run `pnpm run dev` there instead, which proxies /api to this server.
"""

import argparse
import threading
import webbrowser
from pathlib import Path

from dodge import game_settings
from dodge.policies import ARCHITECTURES

DIFFICULTY_NAMES = {1: "easy", 2: "normal", 3: "hard"}
DIFFICULTY_LEVELS = {name: level for level, name in DIFFICULTY_NAMES.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=250_000,
                        help="PPO timesteps between automatic saves over the active model")
    parser.add_argument("--device", default="auto",
                        help="torch device for the policy update: auto, cpu or cuda")
    parser.add_argument("--envs", type=int, default=8,
                        help="games stepped together as lanes of one Rust batch (default 8)")
    parser.add_argument("--rollout-steps", type=int, default=256,
                        help="steps per env per PPO update; the buffer holds "
                             "envs * rollout-steps samples (default 8 * 256)")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default=None,
                        help="start a new model with this architecture; when "
                             "omitted, wait for a checkpoint to be loaded")
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    # Game settings default to whatever the dashboard last saved.
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTY_LEVELS))
    parser.add_argument("--patterns", action=argparse.BooleanOptionalAction)
    parser.add_argument("--powerups", action=argparse.BooleanOptionalAction)
    parser.add_argument("--normal-enemies", action=argparse.BooleanOptionalAction)
    parser.add_argument("--kamikaze-enemies", action=argparse.BooleanOptionalAction)
    parser.add_argument("--spawn-rate-percent", type=int)
    parser.add_argument("--action-repeat", type=int)
    parser.add_argument("--lives", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the dashboard in a browser")
    saved = game_settings.load()
    parser.set_defaults(difficulty=DIFFICULTY_NAMES[saved["difficulty"]],
                        **{key: value for key, value in saved.items() if key != "difficulty"})
    args = parser.parse_args(argv)

    import uvicorn
    from dodge.dashboard import STATIC_DIR, create_app
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(
        seed=args.seed,
        total_steps=args.steps,
        n_envs=args.envs,
        rollout_steps=args.rollout_steps,
        model_dir=args.models_dir,
        difficulty=DIFFICULTY_LEVELS[args.difficulty],
        patterns=args.patterns,
        powerups=args.powerups,
        normal_enemies=args.normal_enemies,
        kamikaze_enemies=args.kamikaze_enemies,
        spawn_rate_percent=args.spawn_rate_percent,
        action_repeat=args.action_repeat,
        lives=args.lives,
        architecture=args.architecture,
        device=args.device,
    )
    url = f"http://{args.host}:{args.port}"
    if not STATIC_DIR.exists():
        print("web/dist is missing: build it with `pnpm install && pnpm run build` in web/")
    print(f"Dashboard at {url}")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    try:
        uvicorn.run(create_app(session), host=args.host, port=args.port, log_level="warning")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
