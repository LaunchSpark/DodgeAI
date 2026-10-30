"""Reference outputs of the velocity-flow-v2 policy, for the parity test.

The speed work on the policy is only allowed to change how fast it runs, not
what it computes. This module records, for fixed observations and fixed seeds,
everything PPO reads from the policy -- logits, values, log-probs, entropy,
sampled actions, and the gradient and Adam step of a PPO-shaped loss -- so a
later build can be compared against the one that recorded it.

Recording is a deliberate act, done once, before an optimisation:

    python -m tests.policy_parity

writes tests/data/policy_parity.npz. Re-recording after a change would make
the test compare the change against itself, so don't.
"""

import copy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "data" / "policy_parity.npz"
CHECKPOINT = ROOT / "artifacts" / "models" / "velocity-flow-v2-latest.zip"

SAMPLE_SEEDS = (0, 1, 7)
# Rollout-sized calls, then the whole set at once: a per-batch-size fast path
# (a CUDA graph, say) must give the same answers at every size it is called at.
CHUNKS = (1, 8, 64)
FRESH_SEED = 0
ADAM_STEPS = 3


def real_observations(lanes=8, snapshots=6, gap=7):
    """Observations from live games: patterns and pickups on, random play."""
    from dodge.env import DodgeVecEnv

    env = DodgeVecEnv(n_envs=lanes, seed=123, difficulty=2, patterns=True,
                      powerups=True, action_repeat=4)
    rng = np.random.default_rng(5)
    try:
        obs = env.reset()
        frames = []
        for _ in range(snapshots):
            for _ in range(gap):
                obs, *_ = env.step(rng.integers(0, 9, lanes))
            frames.append(obs.copy())
    finally:
        env.close()
    return np.concatenate(frames).astype(np.float32)


def synthetic_observations(count=16):
    """Edge cases: an empty arena, the player at the walls, dense noise."""
    from dodge.observe import (CELL_COUNT, CELL_TYPES, OBSERVATION_SIZE,
                               PLAYER_VALUES)

    rng = np.random.default_rng(11)
    obs = np.zeros((count, OBSERVATION_SIZE), dtype=np.float32)
    corners = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.5, 0.5)]
    for i in range(count):
        px, py = corners[i] if i < len(corners) else rng.random(2)
        obs[i, :PLAYER_VALUES] = (px, py, *rng.uniform(-1, 1, PLAYER_VALUES - 2))
        if i < 2:
            continue                       # nothing in the arena at all
        density = 0.02 if i < 8 else 0.3
        occupied = rng.random(CELL_COUNT) < density
        types = rng.integers(1, CELL_TYPES + 1, CELL_COUNT)
        start = PLAYER_VALUES
        obs[i, start:start + CELL_COUNT] = np.where(occupied, types, 0) / CELL_TYPES
        velocity = rng.uniform(-1, 1, (2, CELL_COUNT)) * occupied
        obs[i, start + CELL_COUNT:] = velocity.reshape(-1)
    return obs


def observations():
    return np.concatenate((real_observations(), synthetic_observations()))


def load_policies(device):
    """The shipped checkpoint, and a freshly initialised v2 with a fixed seed."""
    import warnings
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import ConstantSchedule
    from dodge.policies import policy_kwargs
    from dodge.velocity import VelocityFlowPolicy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        latest = PPO.load(CHECKPOINT, device=device).policy
    torch.manual_seed(FRESH_SEED)
    fresh = VelocityFlowPolicy(latest.observation_space, latest.action_space,
                               ConstantSchedule(3e-4),
                               **policy_kwargs("velocity-flow-v2")).to(device)
    return {"latest": latest, "fresh": fresh}


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _projection(policy):
    """A fixed random direction per parameter, to checksum whole tensors."""
    generator = torch.Generator().manual_seed(99)
    return [torch.randn(p.shape, generator=generator) for p in policy.parameters()]


def _loss(policy, obs, actions):
    """A PPO-shaped loss: policy gradient, value regression, entropy bonus."""
    rng = np.random.default_rng(17)
    advantages = torch.as_tensor(rng.standard_normal(len(obs)), dtype=torch.float32,
                                 device=obs.device)
    returns = torch.as_tensor(rng.standard_normal(len(obs)), dtype=torch.float32,
                              device=obs.device)
    values, log_prob, entropy = policy.evaluate_actions(obs, actions)
    return (-(log_prob * advantages).mean()
            + ((values.flatten() - returns) ** 2).mean()
            - 0.01 * entropy.mean())


def outputs(policy, obs_np):
    """Everything recorded for one policy, as a dict of arrays."""
    device = policy.device
    obs = torch.as_tensor(obs_np, device=device)
    actions_fixed = torch.as_tensor(np.arange(len(obs_np)) % 9, device=device)
    result = {}
    policy.set_training_mode(False)
    with torch.no_grad():
        features = policy.extract_features(obs)
        latent_pi, latent_vf = policy.mlp_extractor(features)
        result["features"] = _numpy(features)
        result["logits"] = _numpy(policy.action_net(latent_pi))
        distribution = policy.get_distribution(obs)
        result["log_softmax"] = _numpy(distribution.distribution.logits)
        result["entropy"] = _numpy(distribution.entropy())
        result["values"] = _numpy(policy.predict_values(obs))
        result["deterministic"] = policy.predict(obs_np, deterministic=True)[0]
        for seed in SAMPLE_SEEDS:
            for chunk in CHUNKS:
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                parts = [policy(obs[i:i + chunk]) for i in range(0, len(obs), chunk)]
                actions, values, log_prob = (torch.cat(t) for t in zip(*parts))
                key = f"sample_s{seed}_c{chunk}"
                result[key + "_actions"] = _numpy(actions)
                result[key + "_values"] = _numpy(values)
                result[key + "_log_prob"] = _numpy(log_prob)
        values, log_prob, entropy = policy.evaluate_actions(obs, actions_fixed)
        result["eval_values"] = _numpy(values)
        result["eval_log_prob"] = _numpy(log_prob)
        result["eval_entropy"] = _numpy(entropy)

    # The update path: gradients of a PPO-shaped loss, then a few Adam steps.
    policy.set_training_mode(True)
    projection = _projection(policy)
    policy.zero_grad()
    _loss(policy, obs, actions_fixed).backward()
    grads = [p.grad.detach().cpu() for p in policy.parameters()]
    result["grad_norm"] = np.array([float(g.norm()) for g in grads])
    result["grad_projection"] = np.array(
        [float((g * r).sum()) for g, r in zip(grads, projection)])

    state = {k: v.clone() for k, v in policy.state_dict().items()}
    # Deep-copied: state_dict() hands out the live moment tensors.
    optimizer_state = copy.deepcopy(policy.optimizer.state_dict())
    before = [p.detach().cpu().clone() for p in policy.parameters()]
    for _ in range(ADAM_STEPS):
        policy.optimizer.zero_grad()
        _loss(policy, obs, actions_fixed).backward()
        policy.optimizer.step()
    after = [p.detach().cpu() for p in policy.parameters()]
    result["adam_projection"] = np.array(
        [float(((a - b) * r).sum()) for a, b, r in zip(after, before, projection)])
    result["adam_norm"] = np.array([float((a - b).norm()) for a, b in zip(after, before)])
    # Leave the policy as it was found.
    policy.load_state_dict(state)
    policy.optimizer.load_state_dict(optimizer_state)
    policy.set_training_mode(False)
    return result


def record(path=FIXTURE):
    obs = observations()
    arrays = {"observations": obs}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for name, policy in load_policies(device).items():
            for key, value in outputs(policy, obs).items():
                arrays[f"{device}/{name}/{key}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


if __name__ == "__main__":
    print(f"recorded {record()}")
