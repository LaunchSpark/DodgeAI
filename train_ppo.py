#!/usr/bin/env python3
"""Train a spatial Dodge PPO bot."""

import argparse
from pathlib import Path
import shutil

from contextlib import nullcontext
import math

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file

from dodge.evaluate import preserved_rng
from dodge.lineage import checkpoint_path, jsonable, sidecar_paths
from dodge.policies import (ARCHITECTURES, DEFAULT_ARCHITECTURE,
                            architecture_from_kwargs, policy_class, policy_kwargs)
from dodge.resume import (DEFAULT_SEGMENT_UPDATES, CheckpointWriter,
                          SegmentCallback, load_checkpoint)
from dodge.retention import (RetentionGate, best_path_for,
                             move_legacy_retention)
from dodge.seeds import DEV_COHORT_SIZE, dev_cohort
from dodge.training import (GPU_ARCHITECTURES, SCALE_PRESETS, hyperparameters,
                            make_training_env, minibatch_size,
                            quiet_cpu_policy_warning, resolve_device)

DEFAULT_CHECKPOINT_UPDATES = 5


class ProvenanceCheckpointCallback(BaseCallback):
    """A checkpoint with its state sidecar and manifest every N PPO updates.

    Counted in updates rather than env steps. It used to be SB3's
    CheckpointCallback at save_freq = 10_000 // envs calls -- every ~10k
    decisions, which is every five updates at 8 x 256 but more than once an
    update at 64 x 256 and 256 x 64: a checkpoint file every few seconds, each
    landing mid-rollout. Now it falls just after an update, before the next
    rollout's first step, so it never holds a half-collected rollout and its
    cadence does not move with --envs.

    An update that is also a segment boundary is skipped: the boundary has
    already written the exact checkpoint for it.
    """

    def __init__(self, every_updates, save_path, name_prefix, *, writer, segment):
        super().__init__(verbose=0)
        if every_updates < 0:
            raise ValueError("every_updates cannot be negative")
        self.every_updates = int(every_updates)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.writer = writer
        self.segment = segment
        self.last_saved = None

    def _on_training_start(self):
        # A resumed run starts at the checkpoint's own update; saving that
        # again would only copy the file it was loaded from.
        self.last_saved = self.segment.updates

    def _on_rollout_start(self):
        updates = self.segment.updates
        if (not self.every_updates or updates == 0
                or updates % self.every_updates or updates == self.last_saved
                or self.segment.last_boundary == updates):
            return
        self.writer.save(
            self.model,
            self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps",
            env=self.training_env, ppo_updates=updates,
            segment=self.segment.state(), abandoned=self.segment.abandoned)
        self.last_saved = updates

    def _on_step(self):
        return True


def checkpoint_architecture(path):
    """The architecture a checkpoint holds, read before it is built."""
    data, _, _ = load_from_zip_file(checkpoint_path(path), device="cpu")
    return architecture_from_kwargs(data.get("policy_kwargs"))


def describe_schedule(model, architecture):
    """One line saying what an update costs, since --envs moves all of it."""
    buffer = model.n_envs * model.n_steps
    steps = math.ceil(buffer / model.batch_size) * model.n_epochs
    return (f"{architecture} on {model.device}: {model.n_envs} envs x "
            f"{model.n_steps} steps = {buffer} samples an update, minibatch "
            f"{model.batch_size}, up to {steps} optimizer steps "
            f"({model.n_epochs} epochs, target_kl {model.target_kl})")


def copy_checkpoint(source, target):
    """A checkpoint and its sidecars, byte for byte."""
    for a, b in zip((source, *sidecar_paths(source)), (target, *sidecar_paths(target))):
        if a.exists():
            shutil.copyfile(a, b)


def parse_args(argv=None):
    """The parser and the validated arguments, with --preset applied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=250_000,
                        help="decisions to train for (on --resume, beyond the checkpoint's)")
    parser.add_argument("--device", default="auto",
                        help="torch device for the policy: auto (the GPU for the "
                             "convolutional architectures, if there is one), cpu or cuda")
    parser.add_argument("--preset", choices=tuple(SCALE_PRESETS), default="default",
                        help="env count and rollout length together: 'default' is "
                             "8 x 256; 'gpu' is 64 x 256, the recommended shape on a "
                             "GPU (a 16384-sample buffer in 512-sample minibatches). "
                             "--envs and --rollout-steps override it")
    parser.add_argument("--envs", type=int, default=None,
                        help="games stepped together as lanes of one Rust batch "
                             "(default from --preset: 8)")
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
    parser.add_argument("--rollout-steps", type=int, default=None,
                        help="steps per env per PPO update (default from --preset: 256)")
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
                        help="carry on from the retained best, retained/<name>-best.zip")
    parser.add_argument("--retain-every", type=int, default=0, metavar="N",
                        help="every N PPO updates (a multiple of --segment-updates), "
                             "score the checkpoint on the dev cohort and keep it as "
                             "retained/<name>-best.zip if its mean improves (default off)")
    parser.add_argument("--dev-seeds", type=int, default=DEV_COHORT_SIZE,
                        help=f"dev cohort size for --retain-every (default {DEV_COHORT_SIZE})")
    parser.add_argument("--dev-frame-cap", type=int, default=None,
                        help="frame cap of a dev game (default none: games run until they die)")
    parser.add_argument("--checkpoint-updates", type=int,
                        default=DEFAULT_CHECKPOINT_UPDATES, metavar="N",
                        help="also checkpoint every N PPO updates, between segment "
                             f"boundaries (0 disables; default {DEFAULT_CHECKPOINT_UPDATES})")
    parser.add_argument("--budget-seconds", type=float, default=None,
                        help="stop at the first segment boundary after this much wall "
                             "clock, leaving a resumable checkpoint")
    args = parser.parse_args(argv)
    preset = SCALE_PRESETS[args.preset]
    if args.envs is None:
        args.envs = preset["n_envs"]
    if args.rollout_steps is None:
        args.rollout_steps = preset["rollout_steps"]
    if args.envs < 1:
        parser.error("--envs must be at least 1")
    if args.rollout_steps < 1:
        parser.error("--rollout-steps must be at least 1")
    if args.checkpoint_updates < 0:
        parser.error("--checkpoint-updates cannot be negative")
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
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    model_name = args.name or args.architecture
    resume_path = args.resume
    move_legacy_retention(args.models_dir, model_name)
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
    # "auto" follows the architecture being trained -- the checkpoint's own on
    # a resume -- as the dashboard's does, rather than SB3's "any GPU at all".
    architecture = (checkpoint_architecture(resume_path) if resume_path is not None
                    else args.architecture)
    device = resolve_device(args.device, architecture)
    quiet = (quiet_cpu_policy_warning if architecture in GPU_ARCHITECTURES
             else nullcontext)
    env = make_training_env(args.envs, args.seed, env_kwargs)
    state = None
    if resume_path is not None:
        with quiet():
            model, state = load_checkpoint(
                resume_path, env, device=device,
                custom_objects={"tensorboard_log": str(args.logs_dir), "verbose": 1})
        writer = CheckpointWriter.resuming(resume_path)
    else:
        settings = hyperparameters(args.architecture)
        with quiet():
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
                max_grad_norm=settings["max_grad_norm"],
                verbose=1,
                tensorboard_log=str(args.logs_dir),
                seed=args.seed,
                device=device,
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

    def load_untouched(path):
        # Building a model draws its initial weights from torch's generator.
        with preserved_rng():
            return PPO.load(path, device=model.device)

    hooks = [save_boundary]
    if args.retain_every:
        gate = RetentionGate(best_path_for(args.models_dir, model_name),
                             game_settings=env.game_settings,
                             seeds=dev_cohort(args.dev_seeds),
                             frame_cap=args.dev_frame_cap,
                             load=load_untouched)

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
        args.checkpoint_updates, args.models_dir, model_name,
        writer=writer, segment=segment)
    print(describe_schedule(model, architecture))
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
