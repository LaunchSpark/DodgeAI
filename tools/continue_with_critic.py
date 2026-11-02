#!/usr/bin/env python3
"""Start a velocity-flow-v3 run from a trained velocity-flow-v2 checkpoint.

    python tools/continue_with_critic.py SRC.zip DEST.zip

v3 is v2 with a critic that also reads hidden game state. Every tensor v2 has
is copied across and the critic's new input columns start at zero, so the
result plays and values exactly as v2 did until training moves those columns.
Adam's moments are carried over too, zero-padded where a weight grew, so the
actor keeps stepping as v2 was. A fresh Adam takes larger normalised steps for
its first thousand updates, which would skew a comparison against v2.
"""

import argparse
from pathlib import Path
import sys

# Run directly from anywhere, like the other tools here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from stable_baselines3.common.save_util import load_from_zip_file

from dodge.critic import CRITIC_FEATURE_COUNT
from dodge.env import DodgeVecEnv
from dodge.lineage import checkpoint_path
from dodge.policies import policy_class, policy_kwargs
from dodge.training import (DashboardPPO, checkpoint_architecture, hyperparameters,
                            minibatch_size)

# SB3 registers the shared extractor under several names (features_extractor,
# pi_features_extractor, vf_features_extractor), so this matches by suffix.
WIDENED = "features_extractor.value_net.0.weight"


def _carry_optimizer(old, new):
    """Copy Adam's state and hyperparameters from v2's optimizer to v3's."""
    old_params = dict(old.policy.named_parameters())
    old_optimizer, new_optimizer = old.policy.optimizer, new.policy.optimizer
    for name, parameter in new.policy.named_parameters():
        state = old_optimizer.state.get(old_params[name])
        if not state:
            continue
        carried = {"step": state["step"].clone()}
        for key in ("exp_avg", "exp_avg_sq"):
            moment = torch.zeros_like(parameter)
            if name.endswith(WIDENED):
                moment[:, :state[key].shape[1]] = state[key]
            else:
                moment.copy_(state[key])
            carried[key] = moment
        new_optimizer.state[parameter] = carried
    for key, value in old_optimizer.param_groups[0].items():
        if key != "params":
            new_optimizer.param_groups[0][key] = value


def continue_with_critic(source, dest):
    source = checkpoint_path(source)
    architecture = checkpoint_architecture(source)
    if architecture != "velocity-flow-v2":
        raise ValueError(f"{source} is {architecture}; only velocity-flow-v2 "
                         "can be continued with a critic")
    old = DashboardPPO.load(source, device="cpu")
    _, params, _ = load_from_zip_file(source, device="cpu")

    settings = hyperparameters("velocity-flow-v3")
    env = DodgeVecEnv(n_envs=1, max_episode_frames=20, critic_features=True)
    try:
        model = DashboardPPO(
            policy_class("velocity-flow-v3"), env,
            n_steps=64, batch_size=minibatch_size(64, "velocity-flow-v3"),
            n_epochs=settings["n_epochs"], gamma=old.gamma,
            vf_coef=settings["vf_coef"], target_kl=settings["target_kl"],
            ent_coef=settings["ent_coef"], seed=0, device="cpu",
            policy_kwargs=policy_kwargs("velocity-flow-v3"),
        )
    finally:
        env.close()
    if isinstance(old.learning_rate, (int, float)):
        model.learning_rate = float(old.learning_rate)
        model.lr_schedule = old.lr_schedule

    state = model.policy.state_dict()
    missing = [name for name in state if name not in params["policy"]]
    if missing:
        raise ValueError(f"{source} has no tensor for {missing[0]}")
    with torch.no_grad():
        for name, tensor in params["policy"].items():
            if name not in state:
                raise ValueError(f"{source} has {name}, which v3 does not")
            if name.endswith(WIDENED):
                if tensor.shape[1] + CRITIC_FEATURE_COUNT != state[name].shape[1]:
                    raise ValueError(f"{name} is {tuple(tensor.shape)} in the source")
                state[name].zero_()
                state[name][:, :tensor.shape[1]] = tensor
            elif state[name].shape != tensor.shape:
                raise ValueError(f"{name}: {tuple(tensor.shape)} in the source, "
                                 f"{tuple(state[name].shape)} in v3")
            else:
                state[name].copy_(tensor)

    _carry_optimizer(old, model)
    model.num_timesteps = old.num_timesteps
    model._n_updates = old._n_updates
    model.completed_updates = getattr(old, "completed_updates", None)
    model.save(dest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args(argv)
    continue_with_critic(args.source, args.dest)
    print(f"Wrote {checkpoint_path(args.dest)} from {checkpoint_path(args.source)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
