import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gymnasium import spaces
from stable_baselines3 import PPO

from dodge.observe import (
    CELL_COUNT,
    CELL_TYPES,
    GRID_SIZE,
    HAZARD_LETHAL,
    OBSERVATION_SIZE,
    PLAYER_VALUES,
    VELOCITY_SCALE,
)
from dodge.policies import (
    HAZARD_CHANNELS,
    HYBRID_INPUT_CHANNELS,
    PLAYER_CHANNELS,
    DodgeHybridExtractor,
    architecture_name,
    policy_kwargs,
)

CELLS = PLAYER_VALUES
VX = PLAYER_VALUES + CELL_COUNT
VY = PLAYER_VALUES + 2 * CELL_COUNT
VELOCITY_CHANNEL = HAZARD_CHANNELS + PLAYER_CHANNELS


def observation_space():
    return spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), dtype=np.float32)


def blank():
    return torch.zeros(1, OBSERVATION_SIZE)


def place(observation, row, column, value, velocity=(0.0, 0.0)):
    """Put an entity of ``value`` in a cell, with a velocity in px per frame."""
    index = row * GRID_SIZE + column
    observation[0, CELLS + index] = value / CELL_TYPES
    observation[0, VX + index] = velocity[0] / VELOCITY_SCALE
    observation[0, VY + index] = velocity[1] / VELOCITY_SCALE


class StaticEnv(gymnasium.Env):
    action_space = spaces.Discrete(9)

    def __init__(self):
        self.observation_space = observation_space()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((OBSERVATION_SIZE,), dtype=np.float32), {}

    def step(self, action):
        return (np.zeros((OBSERVATION_SIZE,), dtype=np.float32),
                0.0, False, False, {})


def test_policy_inputs_decode_one_frame_into_eleven_channels():
    """Eight entity types, the player's cell, then a velocity plane per axis."""
    observation = blank()
    observation[0, 0:2] = torch.tensor((0.25, 0.75))
    observation[0, 2:4] = torch.tensor((0.5, -0.5))
    place(observation, 1, 1, HAZARD_LETHAL, velocity=(2.0, -1.0))

    coordinates, grids = DodgeHybridExtractor.policy_inputs(observation)

    assert coordinates.shape == (1, PLAYER_VALUES)
    torch.testing.assert_close(coordinates,
                               torch.tensor(((0.25, 0.75, 0.5, -0.5),)))
    assert grids.shape == (1, HYBRID_INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)
    assert grids[0, HAZARD_LETHAL - 1, 1, 1] == 1
    assert grids[:, :HAZARD_CHANNELS].sum() == 1

    assert grids[0, VELOCITY_CHANNEL, 1, 1] == pytest.approx(2.0 / VELOCITY_SCALE)
    assert grids[0, VELOCITY_CHANNEL + 1, 1, 1] == pytest.approx(-1.0 / VELOCITY_SCALE)
    assert grids[0, VELOCITY_CHANNEL].abs().sum() == pytest.approx(2.0 / VELOCITY_SCALE)


def test_hybrid_extractor_emits_combined_160_features():
    extractor = DodgeHybridExtractor(observation_space())
    output = extractor(torch.zeros((3, OBSERVATION_SIZE)))
    assert output.shape == (3, 160)
    assert torch.isfinite(output).all()


def test_player_channel_marks_the_players_own_cell():
    observation = blank()
    observation[0, 0] = observation[0, 1] = 64.0 / 127.0

    _, grids = DodgeHybridExtractor.policy_inputs(observation)
    player = grids[0, HAZARD_CHANNELS]
    assert player.sum() == 1
    assert player[16, 16] == 1.0


def test_player_channel_follows_the_player_and_clamps_at_the_edge():
    def cell_for(x, y):
        observation = blank()
        observation[0, 0] = x / 127.0
        observation[0, 1] = y / 127.0
        _, grids = DodgeHybridExtractor.policy_inputs(observation)
        return tuple(torch.nonzero(grids[0, HAZARD_CHANNELS])[0].tolist())

    assert cell_for(0, 0) == (0, 0)
    assert cell_for(127, 127) == (GRID_SIZE - 1, GRID_SIZE - 1)
    # row is y, column is x -- a transposition is invisible on the diagonal.
    assert cell_for(100, 20) == (5, 25)


def test_closing_speed_is_the_real_relative_velocity():
    """Not the change in distance between two quantised frames.

    The game knows exactly how fast a threat is approaching. This reads that
    off the velocity plane and projects it onto the line to the threat, so it
    is exact rather than a difference of two rounded positions.
    """
    from dodge.observe import ENEMY_NORMAL
    from dodge.policies import _sense_features

    def closing_for(threat_velocity, player_velocity=(0.0, 0.0)):
        observation = blank()
        observation[0, 0] = observation[0, 1] = 64.0 / 127.0
        observation[0, 2] = player_velocity[0] / VELOCITY_SCALE
        observation[0, 3] = player_velocity[1] / VELOCITY_SCALE
        place(observation, 16, 20, ENEMY_NORMAL, velocity=threat_velocity)
        coordinates, grids = DodgeHybridExtractor.policy_inputs(observation)
        return float(_sense_features(coordinates, grids)[0][5])

    approaching = closing_for((-2.0, 0.0))
    receding = closing_for((2.0, 0.0))
    assert approaching > 0, "moving toward the player closes the gap"
    assert receding < 0
    assert approaching == pytest.approx(-receding)

    # Sideways motion neither closes the gap nor opens it.
    assert closing_for((0.0, 2.0)) == pytest.approx(0.0, abs=1e-6)
    # Chasing a stationary threat is the same as it approaching.
    assert closing_for((0.0, 0.0), player_velocity=(2.0, 0.0)) > 0


def test_nearest_threat_geometry():
    from dodge.observe import ENEMY_NORMAL
    from dodge.policies import _sense_features

    observation = blank()
    observation[0, 0] = observation[0, 1] = 64.0 / 127.0
    place(observation, 16, 19, ENEMY_NORMAL)

    coordinates, grids = DodgeHybridExtractor.policy_inputs(observation)
    _, _, nx, ny, distance, _ = _sense_features(coordinates, grids)[0]

    assert nx * GRID_SIZE == pytest.approx(3.0)
    assert ny * GRID_SIZE == pytest.approx(0.0)
    assert distance * GRID_SIZE == pytest.approx(3.0)


def test_velocity_comes_straight_from_the_observation():
    """No second frame to derive it from, and none needed."""
    from dodge.policies import _sense_features

    observation = blank()
    observation[0, 0] = observation[0, 1] = 64.0 / 127.0
    observation[0, 2] = 2.5 / VELOCITY_SCALE

    coordinates, grids = DodgeHybridExtractor.policy_inputs(observation)
    vx, vy = _sense_features(coordinates, grids)[0][:2]
    assert vx == pytest.approx(2.5 / VELOCITY_SCALE)
    assert vy == pytest.approx(0.0)


def test_pickups_are_not_treated_as_threats():
    """A pickup occupies a cell but cannot end the episode."""
    from dodge.observe import PICKUP_FREEZE
    from dodge.policies import FAR_AWAY, _sense_features

    observation = blank()
    observation[0, 0] = observation[0, 1] = 64.0 / 127.0
    place(observation, 16, 19, PICKUP_FREEZE, velocity=(-2.0, 0.0))

    coordinates, grids = DodgeHybridExtractor.policy_inputs(observation)
    senses = _sense_features(coordinates, grids)[0]
    assert senses[4] == pytest.approx(FAR_AWAY), "no threat in range"
    assert senses[5] == pytest.approx(0.0), "nothing closing"


def test_hybrid_ppo_has_spatial_heads_and_survives_save_load(tmp_path):
    model = PPO(
        "MlpPolicy", StaticEnv(), n_steps=8, batch_size=8, n_epochs=1,
        policy_kwargs=policy_kwargs("hybrid-cnn"), seed=4, device="cpu",
    )
    assert architecture_name(model) == "hybrid-cnn"
    assert model.policy.mlp_extractor.policy_net[0].in_features == 160
    assert model.policy.action_net.out_features == 9

    path = tmp_path / "hybrid-cnn"
    model.save(path)
    restored = PPO.load(path, device="cpu")
    action, _ = restored.predict(np.zeros((OBSERVATION_SIZE,), dtype=np.float32))
    assert architecture_name(restored) == "hybrid-cnn"
    assert 0 <= int(action) < 9


def test_dead_unit_fraction_finds_a_collapsed_policy_trunk():
    """The failure this exists to catch: a policy that ignores its input.

    A trunk whose units are all zero makes action_net emit its bias and
    nothing else, so the agent plays one fixed distribution forever. Entropy
    stays high throughout, which is why it needs its own measurement.
    """
    from dodge.policies import dead_unit_fraction

    model = PPO("MlpPolicy", StaticEnv(), n_steps=8, batch_size=8, n_epochs=1,
                policy_kwargs=policy_kwargs("hybrid-cnn"), seed=1, device="cpu")
    observations = torch.rand(64, OBSERVATION_SIZE)

    assert dead_unit_fraction(model.policy, observations) < 0.5

    # Drive every trunk output negative; LeakyReLU keeps them non-zero, so
    # force the layer to a hard zero the way a plain ReLU would.
    trunk = model.policy.mlp_extractor.policy_net
    with torch.no_grad():
        trunk[-2].weight.zero_()
        trunk[-2].bias.fill_(-50.0)
    trunk[-1] = torch.nn.ReLU()
    assert dead_unit_fraction(model.policy, observations) == 1.0

    with torch.no_grad():
        probabilities = model.policy.get_distribution(observations).distribution.probs
    assert float(probabilities.std(dim=0).max()) == 0.0, (
        "a dead trunk must show up as an input-independent policy"
    )


def test_activations_leak_so_a_unit_cannot_die():
    """A plain ReLU unit that is negative everywhere never gets gradient again."""
    from dodge.policies import activation

    values = torch.tensor([[-5.0, -0.5, 0.0, 2.0]], requires_grad=True)
    output = activation()(values)
    output.sum().backward()

    assert (output[0, :2] != 0).all(), "negative inputs must stay non-zero"
    assert (values.grad != 0).all(), "every unit must keep a live gradient"
    assert output[0, 3] == 2.0, "positive inputs pass through unchanged"
