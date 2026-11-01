import numpy as np

from tools.critic_probe import discounted_returns, gae_returns, explained_variance, fit_probes, probe_r2


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


def test_planted_signal_survives_standardisation_and_early_stopping():
    train, test = _planted(4000, 0), _planted(1000, 1)
    train["x"] = train["x"] * 1000 + 5  # scale must not matter once standardised
    test["x"] = test["x"] * 1000 + 5
    train["x"][:, 0] = 7.0  # constant column becomes 0
    train["lane"] = np.arange(4000) % 10
    base = explained_variance(test["g"], test["v"])
    mlp, ridge = fit_probes(train, test, slice(8, 14), stride=2)
    assert mlp - base > 0.5 and ridge - base > 0.5


def test_gae_returns_hand_computed():
    r = np.array([[1.0], [1.0], [1.0]])
    d = np.array([[0], [1], [0]])
    v = np.full((3, 1), 0.5)
    g = gae_returns(r, d, v, np.array([2.0]), 0.5, 0.5)
    # adv2 = 1 + 0.5*2 - 0.5 = 1.5; adv1 = 1 - 0.5 = 0.5 (terminal);
    # adv0 = (1 + 0.5*0.5 - 0.5) + 0.5*0.5*0.5 = 0.875
    assert np.allclose(g[:, 0], [1.375, 1.0, 2.0])
