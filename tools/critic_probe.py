"""Probe whether critic-only game state predicts what the critic gets wrong.

Plays a model on the wide env, recording its values V, rewards, dones and the
critic-only features. Two targets are scored, each in its own table: the GAE
lambda-return (V + A, gamma and lambda 0.95, bootstrapped past the record; the
target the critic trains on) and the Monte Carlo return (tail without a full
horizon dropped). For each feature group (and all of them) two probes predict
target - V from the standardised [V, features], fitted on a random 1/stride
subset of 3/4 of the lanes and scored on the other lanes: a 2-layer MLP with
early stopping, and a closed-form ridge regression (alpha chosen on held-out
training lanes). Gains are measured against the same probes on V alone.

    python tools/critic_probe.py MODEL.zip --gamma 0.997

Defaults: --envs 1024 --steps 1500 --stride 8 --lam 0.95.
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


def gae_returns(rewards, dones, values, last_values, gamma, lam):
    """GAE lambda-return V + A per step; last_values bootstraps past the record."""
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones).astype(bool)
    values = np.asarray(values, dtype=np.float64)
    adv = np.zeros_like(rewards)
    nxt_adv = np.zeros(rewards.shape[1:])
    nxt_v = np.asarray(last_values, dtype=np.float64)
    for t in range(rewards.shape[0] - 1, -1, -1):
        alive = ~dones[t]
        delta = rewards[t] + gamma * nxt_v * alive - values[t]
        nxt_adv = delta + gamma * lam * alive * nxt_adv
        adv[t] = nxt_adv
        nxt_v = values[t]
    return values + adv


def explained_variance(g, pred):
    return float(1 - np.var(g - pred) / np.var(g))


def _standardise(train_x, *others):
    """Standardise columns with more than 2 distinct fit values, then clip to +-5."""
    train_x = np.asarray(train_x, dtype=np.float64)
    continuous = np.array([len(np.unique(c)) > 2 for c in train_x.T])
    mu = np.where(continuous, train_x.mean(0), 0.0)
    sd = np.where(continuous, train_x.std(0), 1.0)
    sd = np.where(sd == 0, 1.0, sd)
    return [np.clip((np.asarray(x, dtype=np.float64) - mu) / sd, -5, 5).astype(np.float32) for x in (train_x, *others)]


def _inputs(d, columns):
    parts = [d["v"][:, None]]
    if columns is not None:
        parts.append(d["x"][:, columns])
    return np.concatenate(parts, axis=1)


def _split_fit_val(train, stride, seed=0):
    """Hold out 20% of the training lanes (or rows, with no lane ids) as validation."""
    n = len(train["g"])
    rng = np.random.default_rng(seed)
    if "lane" in train:
        lanes = np.unique(train["lane"])
        val_lanes = rng.choice(lanes, max(1, len(lanes) // 5), replace=False)
        is_val = np.isin(train["lane"], val_lanes)
    else:
        is_val = rng.random(n) < 0.2
    # A seeded random 1/stride subset: striding the flat time-major rows would
    # keep only every stride-th lane when the lane count divides by stride.
    thin = rng.random(n) < 1.0 / stride
    return np.flatnonzero(~is_val & thin), np.flatnonzero(is_val & thin)


def fit_probes(train, test, columns, stride=1):
    """Test explained variance of G for (MLP, ridge) fitted to G - V.

    train/test hold flat arrays v (n,), g (n,), x (n, features), and for train
    optionally lane (n,) so validation is held out by lane. columns None means
    V alone. The probes train on every `stride`th row only.
    """
    fit, val = _split_fit_val(train, stride)
    xall = _inputs(train, columns)
    xfit, xval, xte = _standardise(xall[fit], xall[val], _inputs(test, columns))
    resid = (train["g"] - train["v"]).astype(np.float64)
    rfit, rval = resid[fit], resid[val]
    ymu, ysd = rfit.mean(), rfit.std() + 1e-6

    torch.manual_seed(0)
    np.random.seed(0)
    net = nn.Sequential(
        nn.Linear(xfit.shape[1], 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(xfit)
    yt = torch.from_numpy(((rfit - ymu) / ysd).astype(np.float32))
    xv = torch.from_numpy(xval)
    yv = torch.from_numpy(((rval - ymu) / ysd).astype(np.float32))
    rng = np.random.default_rng(0)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(50):
        order = torch.from_numpy(rng.permutation(len(xt)))
        for i in range(0, len(order), 256):
            idx = order[i : i + 256]
            loss = ((net(xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vloss = float(((net(xv).squeeze(-1) - yv) ** 2).mean())
        if vloss < best:
            best, stale = vloss, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        mlp_resid = net(torch.from_numpy(xte)).squeeze(-1).numpy() * ysd + ymu
    mlp_ev = explained_variance(test["g"], test["v"] + mlp_resid)

    # Ridge on the same standardised inputs, alpha picked on the validation lanes.
    aug = lambda x: np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1).astype(np.float64)
    a, av, ate = aug(xfit), aug(xval), aug(xte)
    gram, rhs = a.T @ a, a.T @ rfit
    penalty = np.eye(a.shape[1])
    penalty[-1, -1] = 0
    gval = train["g"][val]
    best_ev, best_w = -np.inf, None
    for alpha in (0.1, 1, 10, 100, 1000):
        w = np.linalg.solve(gram + alpha * penalty, rhs)
        ev = explained_variance(gval, train["v"][val] + av @ w)
        if ev > best_ev:
            best_ev, best_w = ev, w
    ridge_ev = explained_variance(test["g"], test["v"] + ate @ best_w)
    return mlp_ev, ridge_ev


def probe_r2(train, test, columns, stride=1):
    """Test explained variance of G from V plus an MLP's guess at G - V."""
    return fit_probes(train, test, columns, stride)[0]


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
    actor = torch.as_tensor(obs[:, :OBSERVATION_SIZE], dtype=torch.float32, device=model.device)
    with torch.no_grad():
        _, last, _ = model.policy(actor)
    return values, rewards, dones, feats, last.cpu().numpy().reshape(-1)


def report(values, rewards, dones, feats, last_values, gamma, args):
    lam = args.lam
    targets = [
        (f"GAE lambda-return (gamma {gamma}, lambda {lam})", gae_returns(rewards, dones, values, last_values, gamma, lam)),
        (f"Monte Carlo return (gamma {gamma}, NaN tail dropped)", discounted_returns(rewards, dones, gamma)),
    ]
    split = args.envs * 3 // 4
    print(
        f"gamma {gamma}  train lanes {split}  test lanes {args.envs - split}  "
        f"games ended in record {int(dones.sum())}  stride {args.stride}"
    )
    for title, g in targets:

        def part(lanes):
            keep = ~np.isnan(g[:, lanes])
            lane_ids = np.broadcast_to(np.arange(args.envs)[lanes], keep.shape)[keep]
            return dict(v=values[:, lanes][keep], g=g[:, lanes][keep], x=feats[:, lanes][keep], lane=lane_ids)

        train, test = part(slice(0, split)), part(slice(split, None))
        raw = explained_variance(test["g"], test["v"])
        base_mlp, base_ridge = fit_probes(train, test, None, args.stride)
        print(f"\n{title}: train samples {len(train['g'])}  test samples {len(test['g'])}")
        print(f"{'group':<18}{'raw-V EV':>10}{'MLP(V)':>9}{'MLP(V+g)':>10}{'gain':>8}{'ridge(V+g)':>12}{'r.gain':>8}")
        rows = [("all", slice(0, CRITIC_FEATURE_COUNT))] + [(n, slice(*c)) for n, c in CRITIC_GROUPS.items()]
        for name, cols in rows:
            mlp, ridge = fit_probes(train, test, cols, args.stride)
            print(
                f"{name:<18}{raw:>10.4f}{base_mlp:>9.4f}{mlp:>10.4f}{mlp - base_mlp:>+8.4f}"
                f"{ridge:>12.4f}{ridge - base_ridge:>+8.4f}"
            )
        print(
            f"{'baseline (V only)':<18}{raw:>10.4f}{base_mlp:>9.4f}{base_mlp:>10.4f}{0:>+8.4f}"
            f"{base_ridge:>12.4f}{0:>+8.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = DashboardPPO.load(args.model, device="auto")
    print(f"model {args.model}  model.gamma {model.gamma}  steps {args.steps}  envs {args.envs}")
    data = collect(model, args.steps, args.envs, args.seed)
    gammas = [model.gamma if args.gamma is None else args.gamma]
    if args.gamma is not None and model.gamma != args.gamma:
        gammas.append(model.gamma)
    for gamma in gammas:
        report(*data, gamma, args)
        print()


if __name__ == "__main__":
    main()
