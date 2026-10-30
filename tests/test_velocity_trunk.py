"""The v2 forward pass reads the 32x32 trunk without building it.

VelocityFlowV2Extractor.forward, at half resolution, replaces the upsampled
trunk, the 32x32 heads, the pools and the grid_sample with fixed linear maps
applied to the 16x16 body (see the comment above _register_trunk_maps). That
is a claim of equality up to float rounding, and these tests hold it:

* each map against the op it stands in for, on random tensors and on points
  in, on the edge of, and outside the arena;
* the whole extractor, and the whole policy's PPO quantities and gradients,
  against tests/velocity_reference.py -- a frozen copy of the route that built
  the trunk, run on the same parameters.

Tolerances are relative to the largest magnitude in each compared tensor,
as in test_policy_parity.py.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from torch import nn
from torch.nn import functional as F

from dodge import speed
from dodge.observe import GRID_SIZE, OBSERVATION_SIZE, PLAYER_VALUES
from tests import policy_parity, velocity_reference as reference

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# (rel, abs), as in test_policy_parity.py: max|new - old| <= rel * max|old| + abs.
# Measured worst cases are an order of magnitude inside each.
MAP_TOLERANCE = (1e-6, 1e-7)
# A 3x3x64 convolution is a 576-term sum; reordered, it rounds differently by
# about 1e-6 of its largest output.
CONV_TOLERANCE = (1e-5, 1e-7)
FEATURE_TOLERANCE = (1e-5, 1e-7)
# Log-probs pass through _FieldLogits, which divides the nine dangers by their
# spread and so magnifies rounding (see FORWARD_TOLERANCE in the parity test).
LOG_PROB_TOLERANCE = (3e-5, 1e-6)
GRADIENT_TOLERANCE = (1e-4, 1e-9)


def assert_close(new, old, tolerance, name=""):
    rel, floor = tolerance
    assert new.shape == old.shape, name
    new, old = new.detach().double(), old.detach().double()
    worst = float((new - old).abs().max())
    bound = rel * float(old.abs().max()) + floor
    assert worst <= bound, f"{name}: off by {worst:.3g}, allowed {bound:.3g}"


def _space():
    from gymnasium import spaces
    return spaces.Box(low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,),
                      dtype=np.float32)


def _extractor(device, half_resolution=True, seed=0):
    from dodge.velocity import VelocityFlowV2Extractor
    torch.manual_seed(seed)
    return VelocityFlowV2Extractor(_space(),
                                   half_resolution=half_resolution).to(device)


def _random_observations(count=24, seed=4):
    """Uniform noise, players placed anywhere -- including past the walls,
    which puts every path's points outside the arena and onto the border."""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1, 1, (count, OBSERVATION_SIZE)).astype(np.float32)
    obs[:, PLAYER_VALUES:] = np.abs(obs[:, PLAYER_VALUES:])
    obs[:, :2] = rng.uniform(-0.3, 1.3, (count, 2))
    return obs


@pytest.fixture(scope="module")
def observations():
    """Live-game observations, the parity test's edge cases, and noise."""
    return np.concatenate((policy_parity.real_observations(),
                           policy_parity.synthetic_observations(),
                           _random_observations()))


def _edge_points(count, seed=0):
    """Normalised (x, y) points: uniform beyond the arena, plus every edge case
    of align_corners=False -- the outer edges at +-1, cell centres, the
    half-cell borders between the first and last centres and the edge, and
    points a hair either side of each."""
    rng = np.random.default_rng(seed)
    centres = (np.arange(GRID_SIZE) + 0.5) / (GRID_SIZE / 2) - 1
    special = np.concatenate((centres, [-1.0, 1.0, centres[0], centres[-1]]))
    special = np.concatenate((special, special + 1e-6, special - 1e-6,
                              [-1.5, 1.5, 0.0]))
    uniform = rng.uniform(-1.4, 1.4, (count, 2))
    picks = rng.choice(special, (count, 2))
    mixed = np.where(rng.random((count, 2)) < 0.5, picks, uniform)
    # And the corners themselves, exactly.
    mixed[:4] = [(-1, -1), (1, 1), (-1, 1), (centres[-1], centres[0])]
    return torch.as_tensor(mixed, dtype=torch.float32)


def _cells(grid):
    """Normalised grid_sample points in cell units, as grid_sample computes
    them for align_corners=False."""
    return ((grid + 1) * GRID_SIZE - 1) / 2


def _read(field, weights_x, weights_y):
    """Bilinear reads of (n, c, h, w) at (n, p) per-axis weights: (n, c, p)."""
    return torch.einsum("nchw,nph,npw->ncp", field, weights_y, weights_x)


# --- each map against the op it replaces ----------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_cell_weights_read_exactly_what_grid_sample_reads(device):
    extractor = _extractor(device)
    n, points = 6, 300
    field = torch.randn(n, 6, GRID_SIZE, GRID_SIZE, device=device)
    grid = torch.stack([_edge_points(points, seed) for seed in range(n)]).to(device)

    expected = F.grid_sample(field, grid[:, :, None, :], mode="bilinear",
                             padding_mode="border", align_corners=False)[..., 0]
    weights = extractor._cell_weights(_cells(grid))
    assert weights.shape == (n, points, 2, GRID_SIZE)
    # Interpolation weights: non-negative, summing to one at every point.
    assert bool((weights >= 0).all())
    torch.testing.assert_close(weights.sum(-1), torch.ones(n, points, 2, device=device))
    assert_close(_read(field, weights[..., 0, :], weights[..., 1, :]), expected,
                 MAP_TOLERANCE, "grid_sample")


@pytest.mark.parametrize("device", DEVICES)
def test_reads_through_the_basis_match_grid_sample_on_the_upsampled_field(device):
    """A point read of upsample(body) + upsample(context) is a read of body and
    of context, with the 32-cell weights carried back to 16 and to 4."""
    extractor = _extractor(device)
    n, points, half = 5, 200, GRID_SIZE // 2
    body = torch.randn(n, 6, half, half, device=device)
    context = torch.randn(n, 6, 4, 4, device=device)
    size = (GRID_SIZE, GRID_SIZE)
    field = (F.interpolate(body, size=size, mode="bilinear", align_corners=False)
             + F.interpolate(context, size=size, mode="bilinear", align_corners=False))
    grid = torch.stack([_edge_points(points, seed + 10) for seed in range(n)]).to(device)

    expected = F.grid_sample(field, grid[:, :, None, :], mode="bilinear",
                             padding_mode="border", align_corners=False)[..., 0]
    weights = extractor._cell_weights(_cells(grid)) @ extractor.read_basis
    got = (_read(body, weights[..., 0, :half], weights[..., 1, :half])
           + _read(context, weights[..., 0, half:], weights[..., 1, half:]))
    assert_close(got, expected, MAP_TOLERANCE, "upsampled read")


@pytest.mark.parametrize("device", DEVICES)
def test_trunk_pool_is_the_adaptive_pool_of_the_upsampled_body(device):
    from dodge.velocity import CONTEXT_GRID

    extractor = _extractor(device)
    body = torch.randn(7, 32, GRID_SIZE // 2, GRID_SIZE // 2, device=device)
    trunk = F.interpolate(body, size=(GRID_SIZE, GRID_SIZE), mode="bilinear",
                          align_corners=False)
    pooled = F.linear(body.flatten(2), extractor.trunk_pool).unflatten(
        2, (CONTEXT_GRID, CONTEXT_GRID))
    assert_close(pooled, F.adaptive_avg_pool2d(trunk, CONTEXT_GRID),
                 MAP_TOLERANCE, "pool")
    # And the means the critic reads.
    assert_close(pooled.mean(dim=(2, 3)), trunk.mean(dim=(2, 3)), MAP_TOLERANCE, "mean")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", [torch.contiguous_format, torch.channels_last])
def test_parity_split_and_block_mean_are_the_ops_they_replace(device, layout):
    """Run on every device, though only CUDA takes these routes in training."""
    torch.manual_seed(2)
    conv = nn.Conv2d(64, 64, 3, padding=2, dilation=2).to(device)
    x = torch.randn(9, 64, 16, 16, device=device).contiguous(memory_format=layout)
    expected = F.conv2d(x, conv.weight, conv.bias, padding=2, dilation=2)
    assert_close(speed.parity_conv(conv, x), expected, CONV_TOLERANCE, "parity conv")
    assert_close(speed.dilated_conv(conv, x), expected, CONV_TOLERANCE, "dilated conv")

    x = torch.randn(9, 32, 32, 32, device=device).contiguous(memory_format=layout)
    assert_close(speed.block_mean(x), F.avg_pool2d(x, 2), MAP_TOLERANCE, "block mean")
    assert_close(speed.halve(x), F.avg_pool2d(x, 2), MAP_TOLERANCE, "halve")


def test_checkpoints_see_the_same_parameters():
    """Same names, shapes and module types where a checkpoint can tell; the
    derived maps are not saved."""
    extractor = _extractor("cpu")
    keys = extractor.state_dict().keys()
    assert not {"trunk_pool", "read_basis", "cell_index"} & set(keys)
    dilated = extractor.field_net.body[2]
    assert isinstance(dilated, nn.Conv2d)
    assert (dilated.kernel_size, dilated.padding, dilated.dilation) == ((3, 3), (2, 2), (2, 2))
    body = {k: tuple(v.shape) for k, v in extractor.state_dict().items()
            if k.startswith("field_net.")}
    assert body == {
        "field_net.stem.0.weight": (32, 11, 3, 3), "field_net.stem.0.bias": (32,),
        "field_net.body.0.weight": (64, 32, 3, 3), "field_net.body.0.bias": (64,),
        "field_net.body.2.weight": (64, 64, 3, 3), "field_net.body.2.bias": (64,),
        "field_net.body.4.weight": (32, 64, 3, 3), "field_net.body.4.bias": (32,),
    }


# --- the whole extractor and policy against the frozen route --------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("half_resolution", [True, False])
def test_extractor_matches_the_frozen_route(device, half_resolution, observations):
    extractor = _extractor(device, half_resolution)
    obs = torch.as_tensor(observations, device=device)
    with torch.no_grad():
        assert_close(extractor(obs), reference.forward(extractor, obs),
                     FEATURE_TOLERANCE, "features")
        # field() still builds the whole field, for the overlay.
        assert_close(extractor.field(obs), reference.field(extractor, obs),
                     FEATURE_TOLERANCE, "field")


@pytest.mark.parametrize("device", DEVICES)
def test_v1_extractor_matches_the_frozen_route(device, observations):
    from stable_baselines3.common.utils import ConstantSchedule
    from dodge.policies import policy_kwargs
    from dodge.velocity import VelocityFlowPolicy

    torch.manual_seed(0)
    policy = VelocityFlowPolicy(_space(), _action_space(), ConstantSchedule(3e-4),
                                **policy_kwargs("velocity-flow")).to(device)
    extractor = policy.features_extractor
    obs = torch.as_tensor(observations, device=device)
    with torch.no_grad():
        assert_close(extractor(obs), reference.forward(extractor, obs),
                     FEATURE_TOLERANCE, "features")


def _action_space():
    from gymnasium import spaces
    return spaces.Discrete(9)


def _policies(device):
    """A fresh seeded v2 policy, and the shipped checkpoint when present."""
    from stable_baselines3.common.utils import ConstantSchedule
    from dodge.policies import policy_kwargs
    from dodge.velocity import VelocityFlowPolicy

    torch.manual_seed(policy_parity.FRESH_SEED)
    policies = {"fresh": VelocityFlowPolicy(
        _space(), _action_space(), ConstantSchedule(3e-4),
        **policy_kwargs("velocity-flow-v2")).to(device)}
    if policy_parity.CHECKPOINT.exists():
        policies["latest"] = policy_parity.load_policies(device)["latest"]
    return policies


def _ppo_quantities(policy, obs, actions):
    """Values, log-probs, entropy, and every parameter's gradient of a PPO loss."""
    policy.set_training_mode(True)
    policy.zero_grad()
    values, log_prob, entropy = policy.evaluate_actions(obs, actions)
    rng = np.random.default_rng(17)
    advantages = torch.as_tensor(rng.standard_normal(len(obs)), dtype=torch.float32,
                                 device=obs.device)
    returns = torch.as_tensor(rng.standard_normal(len(obs)), dtype=torch.float32,
                              device=obs.device)
    loss = (-(log_prob * advantages).mean()
            + ((values.flatten() - returns) ** 2).mean() - 0.01 * entropy.mean())
    loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in policy.named_parameters()}
    policy.zero_grad()
    policy.set_training_mode(False)
    return values.detach(), log_prob.detach(), entropy.detach(), grads


def _swapped(extractor):
    """The extractor with its forward replaced by the frozen route, for a
    with-block: every other op, and every parameter, stays shared."""
    class Swap:
        def __enter__(self):
            extractor.forward = lambda o: reference.forward(extractor, o)

        def __exit__(self, *_):
            del extractor.forward
    return Swap()


def _assert_gradients_close(new, old, tolerance, name):
    """Per tensor, by norm -- as the parity test checks them by checksums."""
    assert new.keys() == old.keys()
    for key in new:
        a, b = new[key].double(), old[key].double()
        error = float((a - b).norm()) / (float(b.norm()) + 1e-30)
        assert error <= tolerance, f"{name}/grad {key}: off by {error:.3g} of its norm"


# Gradients are where float32 cannot settle the question. A pre-activation
# within rounding of zero sits on LeakyReLU's kink, where the derivative
# jumps from 0.01 to 1, and any reordering of the sums before it -- this
# change, or cuDNN autotuning picking another kernel for the old route -- can
# round it to the other side. The training gradient is concentrated on the 54
# points each sample reads, so one such cell near a read moves the trunk's
# weight gradients by ~4e-4 of their norm (measured, on the fresh policy, in
# roughly one CUDA process in five). A fresh policy has many: Stable-Baselines3
# zeroes every bias, so empty arena gives pre-activations of exactly 0, which
# a Winograd-style kernel returns as ~1e-9 noise. None of that is a difference
# in what the routes compute, so the strict gradient test runs in float64,
# where it cannot happen; float32 gradients are held only on the trained
# checkpoint, whose activations keep clear of the kink (2e-6, every run).
FLOAT64_GRADIENT_TOLERANCE = 1e-10


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trained", [False, True])
def test_extractor_gradients_match_the_frozen_route_in_float64(device, trained,
                                                               observations):
    """Every parameter's gradient through the full extractor, in float64.

    Stable-Baselines3 casts observations to float32 before the extractor, so
    this drives the extractor directly. Not under PPO's loss: _FieldLogits
    removes the nine dangers' mean, and with it every gradient of the field
    head's bias, which then compares rounding residue with rounding residue.
    A nonlinearity of every feature leaves each parameter a real gradient.
    """
    if trained:
        if not policy_parity.CHECKPOINT.exists():
            pytest.skip("velocity-flow-v2-latest.zip is not present")
        extractor = policy_parity.load_policies(device)["latest"].features_extractor
    else:
        extractor = _extractor(device)
    extractor = extractor.double()
    obs = torch.as_tensor(observations, device=device, dtype=torch.float64)
    weights = torch.randn(len(obs), extractor.features_dim, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(3)).to(device)

    def gradients():
        extractor.zero_grad()
        features = extractor(obs)
        (features.tanh() * weights).sum().backward()
        return features.detach(), {k: p.grad.clone()
                                   for k, p in extractor.named_parameters()}

    try:
        features, new = gradients()
        with _swapped(extractor):
            expected, old = gradients()
    finally:
        extractor.float()
    assert_close(features, expected, (1e-12, 1e-15), "features")
    _assert_gradients_close(new, old, FLOAT64_GRADIENT_TOLERANCE, "float64")


@pytest.mark.parametrize("device", DEVICES)
def test_policy_outputs_and_gradients_match_the_frozen_route(device, observations):
    """What PPO reads, in float32, through the policy itself."""
    obs = torch.as_tensor(observations, device=device)
    actions = torch.arange(len(obs), device=device) % 9
    for name, policy in _policies(device).items():
        new = _ppo_quantities(policy, obs, actions)
        with _swapped(policy.features_extractor):
            old = _ppo_quantities(policy, obs, actions)
        for label, a, b, tolerance in (
                ("values", new[0], old[0], FEATURE_TOLERANCE),
                ("log_prob", new[1], old[1], LOG_PROB_TOLERANCE),
                ("entropy", new[2], old[2], LOG_PROB_TOLERANCE)):
            assert_close(a, b, tolerance, f"{name}/{label}")
        if name == "latest":                 # see FLOAT64_GRADIENT_TOLERANCE
            _assert_gradients_close(new[3], old[3], GRADIENT_TOLERANCE[0], name)
