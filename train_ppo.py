#!/usr/bin/env python3
"""Train a spatial Dodge PPO bot."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from dodge.env import DodgeEnv
from dodge.policies import (ARCHITECTURES, DEFAULT_ARCHITECTURE,
                            policy_class, policy_kwargs)
from dodge.training import hyperparameters, minibatch_size


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=250_000)
    parser.add_argument("--device", default="auto",
                        help="torch device for the policy update")
    parser.add_argument("--envs", type=int, default=8,
                        help="games stepped in parallel processes (default 8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=("easy", "normal", "hard"), default="easy")
    parser.add_argument("--patterns", action="store_true")
    parser.add_argument("--powerups", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--normal-enemies", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--kamikaze-enemies", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--spawn-rate-percent", type=int, default=100)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--architecture", choices=ARCHITECTURES,
                        default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--name", default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--logs-dir", type=Path, default=Path("artifacts/tensorboard"))
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args(argv)
    model_name = args.name or args.architecture

    difficulty = {"easy": 1, "normal": 2, "hard": 3}[args.difficulty]
    env_kwargs = {
        "difficulty": difficulty,
        "patterns": args.patterns,
        "powerups": args.powerups,
        "normal_enemies": args.normal_enemies,
        "kamikaze_enemies": args.kamikaze_enemies,
        "spawn_rate_percent": args.spawn_rate_percent,
        "action_repeat": args.action_repeat,
    }
    if args.check_env:
        check_env(DodgeEnv(**env_kwargs))

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    # Separate processes are what make extra envs faster; DummyVecEnv would
    # step each Dodge machine sequentially in this one.
    env = make_vec_env(DodgeEnv, n_envs=args.envs, seed=args.seed,
                       env_kwargs=env_kwargs,
                       vec_env_cls=SubprocVecEnv if args.envs > 1 else DummyVecEnv)
    checkpoint = CheckpointCallback(
        save_freq=max(10_000 // args.envs, 1),
        save_path=str(args.models_dir),
        name_prefix=model_name,
    )
    settings = hyperparameters(args.architecture)
    model = PPO(
        policy_class(args.architecture),
        env,
        learning_rate=3e-4,
        n_steps=args.rollout_steps,
        # From the architecture's table, not hardcoded. These were fixed at
        # v1's schedule, so --architecture velocity-flow-v2 here built v2's
        # network and then trained it on ten epochs at half the minibatch.
        batch_size=minibatch_size(args.envs * args.rollout_steps,
                                  args.architecture),
        n_epochs=settings["n_epochs"],
        gamma=settings["gamma"],
        vf_coef=settings["vf_coef"],
        target_kl=settings["target_kl"],
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=settings["ent_coef"],
        verbose=1,
        tensorboard_log=str(args.logs_dir),
        seed=args.seed,
        device=args.device,
        policy_kwargs=policy_kwargs(args.architecture),
    )
    try:
        model.learn(total_timesteps=args.steps, callback=checkpoint,
                    tb_log_name=model_name)
        output = args.models_dir / model_name
        model.save(output)
        print(f"Saved PPO model to {output}.zip")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
