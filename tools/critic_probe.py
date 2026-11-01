"""Probe whether critic-only game state predicts what the critic gets wrong.

Plays a model on the wide env, records its values V and the realised
discounted returns G, then fits a small MLP on [V, features] to predict G - V
and reports held-out explained variance of G with and without it.

    python tools/critic_probe.py MODEL.zip --steps 20000 --envs 32 --gamma 0.997
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dodge.critic import CRITIC_FEATURE_COUNT, CRITIC_GROUPS  # noqa: E402
from dodge.env import DodgeVecEnv  # noqa: E402
from dodge.observe import OBSERVATION_SIZE  # noqa: E402
from dodge.training import DashboardPPO  # noqa: E402


def discounted_returns(rewards, dones, gamma):
    """Per-lane discounted return; a done step ends its return.

    NaN where no terminal lies at or after the step and fewer than
    ceil(5 / (1 - gamma)) steps remain (the return is not yet settled).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones).astype(bool)
    steps = rewards.shape[0]
    horizon = math.ceil(5 / (1 - gamma))
    g = np.zeros_like(rewards)
    nxt = np.zeros(rewards.shape[1:])
    terminal_ahead = np.zeros(rewards.shape[1:], dtype=bool)
    for t in range(steps - 1, -1, -1):
        nxt = rewards[t] + gamma * nxt * ~dones[t]
        g[t] = nxt
        terminal_ahead |= dones[t]
        g[t][~terminal_ahead & (steps - t < horizon)] = np.nan
    return g


def explained_variance(g, pred):
    return float(1 - np.var(g - pred) / np.var(g))


def probe_r2(train, test, columns):
    """Test-set explained variance of G from V plus an MLP's guess at G - V.

    train/test hold flat arrays: v (n,), g (n,), x (n, features).
    columns None means V alone goes into the MLP.
    """
    def inputs(d):
        parts = [d["v"][:, None]]
        if columns is not None:
            parts.append(d["x"][:, columns])
        return np.concatenate(parts, axis=1).astype(np.float32)

    xtr, xte = inputs(train), inputs(test)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    ytr = (train["g"] - train["v"]).astype(np.float32)
    ymu, ysd = ytr.mean(), ytr.std() + 1e-6
    ytr = (ytr - ymu) / ysd

    torch.manual_seed(0)
    np.random.seed(0)
    net = nn.Sequential(
        nn.Linear(xtr.shape[1], 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    rng = np.random.default_rng(0)
    for _ in range(30):
        order = torch.from_numpy(rng.permutation(len(xt)))
        for i in range(0, len(order), 256):
            idx = order[i : i + 256]
            loss = ((net(xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        resid = net(torch.from_numpy(xte)).squeeze(-1).numpy() * ysd + ymu
    return explained_variance(test["g"], test["v"] + resid)


def collect(model, steps, envs, seed):
    env = DodgeVecEnv(
        n_envs=envs, seed=seed, critic_features=True, difficulty=3, patterns=True, powerups=True, action_repeat=1
    )
    obs = env.reset()
    values = np.zeros((steps, envs))
    rewards = np.zeros((steps, envs))
    dones = np.zeros((steps, envs))
    feats = np.zeros((steps, envs, CRITIC_FEATURE_COUNT), dtype=np.float32)
    for t in range(steps):
        actor = torch.as_tensor(obs[:, :OBSERVATION_SIZE], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            actions, v, _ = model.policy(actor)
        values[t] = v.cpu().numpy().reshape(-1)
        feats[t] = obs[:, OBSERVATION_SIZE:]
        obs, r, d, _ = env.step(actions.cpu().numpy())
        rewards[t], dones[t] = r, d
    return values, rewards, dones, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = DashboardPPO.load(args.model, device="auto")
    gamma = model.gamma if args.gamma is None else args.gamma
    values, rewards, dones, feats = collect(model, args.steps, args.envs, args.seed)
    g = discounted_returns(rewards, dones, gamma)

    split = args.envs * 3 // 4

    def part(lanes):
        v, gg, x = values[:, lanes], g[:, lanes], feats[:, lanes]
        keep = ~np.isnan(gg)
        return dict(v=v[keep], g=gg[keep], x=x[keep])

    train, test = part(slice(0, split)), part(slice(split, None))
    base = explained_variance(test["g"], test["v"])
    print(f"model {args.model}  gamma {gamma}  steps {args.steps}  envs {args.envs}")
    print(f"train samples {len(train['g'])}  test samples {len(test['g'])}")
    print(f"{'features':<18}{'EV(V+probe)':>12}{'gain':>9}")
    print(f"{'baseline (V only)':<18}{base:>12.4f}{0.0:>9.4f}")
    rows = [("all", slice(0, CRITIC_FEATURE_COUNT))] + list(CRITIC_GROUPS.items())
    for name, (cols) in rows:
        columns = cols if isinstance(cols, slice) else slice(*cols)
        ev = probe_r2(train, test, columns)
        print(f"{name:<18}{ev:>12.4f}{ev - base:>+9.4f}")


if __name__ == "__main__":
    main()
