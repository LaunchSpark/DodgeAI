#!/usr/bin/env python3
"""Train a spatial Dodge PPO bot."""

import argparse
from pathlib import Path
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from dodge.lineage import jsonable, sidecar_paths
from dodge.policies import (ARCHITECTURES, DEFAULT_ARCHITECTURE,
                            policy_class, policy_kwargs)
from dodge.resume import (DEFAULT_SEGMENT_UPDATES, CheckpointWriter,
                          SegmentCallback, load_checkpoint)
from dodge.retention import DEV_FRAME_CAP, RetentionGate, best_path_for
from dodge.seeds import DEV_COHORT_SIZE, dev_cohort
from dodge.training import hyperparameters, make_training_env, minibatch_size


class ProvenanceCheckpointCallback(CheckpointCallback):
    """SB3's periodic checkpoint, with the state sidecar and manifest."""

    def __init__(self, *args, writer, segment, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.segment = segment

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.writer.save(self.model, self._checkpoint_path(extension="zip"),
                             env=self.training_env, ppo_updates=self.segment.updates,
                             segment=self.segment.state(),
                             abandoned=self.segment.abandoned)
        return True


def copy_checkpoint(source, target):
    """A checkpoint and its sidecars, byte for byte."""
    for a, b in zip((source, *sidecar_paths(source)), (target, *sidecar_paths(target))):
        if a.exists():
            shutil.copyfile(a, b)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=250_000,
                        help="decisions to train for (on --resume, beyond the checkpoint's)")
    parser.add_argument("--device", default="auto",
                        help="torch device for the policy update")
    parser.add_argument("--envs", type=int, default=8,
                        help="games stepped together as lanes of one Rust batch (default 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="run seed: model initialisation and the game seed stream")
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
    parser.add_argument("--segment-updates", type=int, default=DEFAULT_SEGMENT_UPDATES,
                        help="abandon every game and checkpoint every N PPO updates, "
                             "so the checkpoint resumes exactly (0 disables; "
                             f"default {DEFAULT_SEGMENT_UPDATES})")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", type=Path, default=None, metavar="PATH",
                        help="carry on from a checkpoint: model, Adam, RNG and seed "
                             "stream (exact from a segment-boundary checkpoint)")
    resume.add_argument("--from-best", action="store_true",
                        help="carry on from the retained best, <name>-best.zip")
    parser.add_argument("--retain-every", type=int, default=0, metavar="N",
                        help="every N PPO updates (a multiple of --segment-updates), "
                             "score the checkpoint on the dev cohort and keep it as "
                             "<name>-best.zip if its mean improves (default off)")
    parser.add_argument("--dev-seeds", type=int, default=DEV_COHORT_SIZE,
                        help=f"dev cohort size for --retain-every (default {DEV_COHORT_SIZE})")
    parser.add_argument("--dev-frame-cap", type=int, default=DEV_FRAME_CAP,
                        help=f"frame cap of a dev game (default {DEV_FRAME_CAP})")
    parser.add_argument("--budget-seconds", type=float, default=None,
                        help="stop at the first segment boundary after this much wall "
                             "clock, leaving a resumable checkpoint")
    args = parser.parse_args(argv)
    if args.segment_updates < 0:
        parser.error("--segment-updates cannot be negative")
    if args.budget_seconds is not None and not args.segment_updates:
        parser.error("--budget-seconds stops at a segment boundary: "
                     "it needs --segment-updates > 0")
    if args.retain_every:
        if not args.segment_updates:
            parser.error("--retain-every scores segment-boundary checkpoints: "
                         "it needs --segment-updates > 0")
        if args.retain_every % args.segment_updates:
            parser.error("--retain-every must be a multiple of --segment-updates")
    model_name = args.name or args.architecture
    resume_path = args.resume
    if args.from_best:
        resume_path = best_path_for(args.models_dir, model_name)
        if not resume_path.exists():
            parser.error(f"no retained best at {resume_path}")

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
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    env = make_training_env(args.envs, args.seed, env_kwargs)
    state = None
    if resume_path is not None:
        model, state = load_checkpoint(
            resume_path, env, device=args.device,
            custom_objects={"tensorboard_log": str(args.logs_dir), "verbose": 1})
        writer = CheckpointWriter.resuming(resume_path)
    else:
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
        writer = CheckpointWriter()
    writer.extra_config = {"cli": jsonable(vars(args)),
                           "resumed_from": str(resume_path) if resume_path else None}

    def save_boundary(callback):
        path = writer.save(callback.model,
                           args.models_dir / f"{model_name}-u{callback.updates:07d}",
                           env=callback.training_env, ppo_updates=callback.updates,
                           at_boundary=True, segment=callback.state(),
                           abandoned=callback.abandoned)
        print(f"Segment boundary at update {callback.updates}: abandoned "
              f"{callback.training_env.num_envs} games, saved {path}")

    hooks = [save_boundary]
    if args.retain_every:
        gate = RetentionGate(best_path_for(args.models_dir, model_name),
                             game_settings=env.game_settings,
                             seeds=dev_cohort(args.dev_seeds),
                             frame_cap=args.dev_frame_cap)

        def retain(callback):
            if callback.updates % args.retain_every:
                return
            outcome = gate.consider(callback.model, writer.last_path,
                                    update=callback.updates)
            verdict = "retained as best" if outcome["promoted"] else "not retained"
            print(f"Dev cohort at update {callback.updates}: mean "
                  f"{outcome['dev']['mean']:.0f}, Q1 {outcome['dev']['q1']:.0f} "
                  f"-- {verdict} (best {outcome['best_mean']:.0f})")
        hooks.append(retain)

    segment = SegmentCallback.from_state(state, args.segment_updates,
                                         on_boundary=hooks,
                                         budget_seconds=args.budget_seconds)
    checkpoint = ProvenanceCheckpointCallback(
        save_freq=max(10_000 // args.envs, 1),
        save_path=str(args.models_dir),
        name_prefix=model_name,
        writer=writer,
        segment=segment,
    )
    try:
        model.learn(total_timesteps=args.steps, callback=[segment, checkpoint],
                    tb_log_name=model_name,
                    reset_num_timesteps=resume_path is None)
        output = args.models_dir / f"{model_name}.zip"
        if segment.stop_requested:
            # The budget ran out at a boundary, and training took one step
            # past it before stopping: the boundary checkpoint is the one that
            # resumes exactly, so it is the run's output.
            copy_checkpoint(writer.last_path, output)
            print(f"Wall-clock budget spent: stopped at update {segment.updates}")
        else:
            writer.save(model, output, env=env, ppo_updates=segment.updates,
                        at_boundary=segment.last_boundary == segment.updates
                        and segment.updates > 0,
                        segment=segment.state(), abandoned=segment.abandoned)
        print(f"Saved PPO model to {output}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
