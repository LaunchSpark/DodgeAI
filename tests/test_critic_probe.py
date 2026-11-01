import numpy as np

from tools.critic_probe import discounted_returns, explained_variance, probe_r2


def test_discounted_returns_stop_at_terminals():
    r = np.array([[1.0], [1.0], [1.0]])
    d = np.array([[0], [1], [0]])
    g = discounted_returns(r, d, 0.5)
    assert g[0, 0] == 1 + 0.5 * 1 and g[1, 0] == 1
    assert np.isnan(g[2, 0])


def _planted(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 20))
    v = rng.normal(size=n)
    g = v + 3 * x[:, 10] + 0.1 * rng.normal(size=n)
    return dict(v=v, g=g, x=x)


def test_probe_finds_a_planted_signal():
    train, test = _planted(4000, 0), _planted(1000, 1)
    base = explained_variance(test["g"], test["v"])
    assert probe_r2(train, test, slice(8, 14)) - base > 0.5
    assert abs(probe_r2(train, test, None) - base) < 0.05
