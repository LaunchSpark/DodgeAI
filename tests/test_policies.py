import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gymnasium import spaces
from stable_baselines3 import PPO

from dodge.observe import (
    CELL_TYPES,
    FRAME_OBSERVATION_SIZE,
    GRID_SIZE,
    HAZARD_LETHAL,
    OBSERVATION_SIZE,
)
from dodge.policies import (
    HAZARD_CHANNELS,
    HYBRID_INPUT_CHANNELS,
    DodgeHybridExtractor,
    architecture_name,
    policy_kwargs,
)


def observation_space():
    return spaces.Box(0.0, 1.0, (OBSERVATION_SIZE,), dtype=np.float32)


def test_hybrid_extractor_builds_one_hot_channels_for_both_frames():
    observation = torch.zeros((1, OBSERVATION_SIZE), dtype=torch.float32)
    observation[0, 0:2] = torch.tensor((0.25, 0.75))
    observation[0, 2] = 1 / CELL_TYPES
    current = FRAME_OBSERVATION_SIZE
    observation[0, current:current + 2] = torch.tensor((0.5, 0.125))
    observation[0, current + 2 + 33] = HAZARD_LETHAL / CELL_TYPES

    coordinates, grids = DodgeHybridExtractor.policy_inputs(observation)

    assert coordinates.shape == (1, 4)
    torch.testing.assert_close(
        coordinates,
        torch.tensor(((0.25, 0.75, 0.5, 0.125),)),
    )
    assert grids.shape == (1, HYBRID_INPUT_CHANNELS, 32, 32)
    assert grids[0, 0, 0, 0] == 1
    assert grids[0, 8 + HAZARD_LETHAL - 1, 1, 1] == 1
    # Two hazard cells plus one player mark per frame.
    assert grids[:, :HAZARD_CHANNELS].sum() == 2
    assert grids[:, HAZARD_CHANNELS:].sum() == 2


def test_hybrid_extractor_emits_combined_160_features():
    extractor = DodgeHybridExtractor(observation_space())
    output = extractor(torch.zeros((3, OBSERVATION_SIZE)))
    assert output.shape == (3, 160)
    assert torch.isfinite(output).all()


def test_hybrid_ppo_has_spatial_heads_and_survives_save_load(tmp_path):
    class StaticEnv(gymnasium.Env):
        observation_space = observation_space()
        action_space = spaces.Discrete(9)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros((OBSERVATION_SIZE,), dtype=np.float32), {}

        def step(self, action):
            return (
                np.zeros((OBSERVATION_SIZE,), dtype=np.float32),
                0.0,
                False,
                False,
                {},
            )

    model = PPO(
        "MlpPolicy",
        StaticEnv(),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs=policy_kwargs("hybrid-cnn"),
        seed=4,
        device="cpu",
    )
    assert architecture_name(model) == "hybrid-cnn"
    assert model.policy.mlp_extractor.policy_net[0].in_features == 160
    assert model.policy.mlp_extractor.policy_net[0].out_features == 128
    assert model.policy.mlp_extractor.policy_net[2].out_features == 64
    assert model.policy.action_net.out_features == 9

    path = tmp_path / "hybrid-cnn"
    model.save(path)
    restored = PPO.load(path, device="cpu")
    action, _ = restored.predict(np.zeros((OBSERVATION_SIZE,), dtype=np.float32))
    assert architecture_name(restored) == "hybrid-cnn"
    assert 0 <= int(action) < 9


def test_player_channels_mark_the_players_own_cell():
    """The hazard map alone never said where the agent was.

    Every convolution now sees the player inside its receptive field, so
    "close to me" is spatial rather than something the coordinate branch has
    to reconstruct after the layout has been flattened.
    """

    observation = torch.zeros(1, OBSERVATION_SIZE)
    # Player at the arena centre in both frames; hazard cells left empty.
    for frame in range(2):
        observation[0, frame * FRAME_OBSERVATION_SIZE] = 64.0 / 127.0
        observation[0, frame * FRAME_OBSERVATION_SIZE + 1] = 64.0 / 127.0

    _, grids = DodgeHybridExtractor.policy_inputs(observation)
    assert grids.shape == (1, HYBRID_INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)

    player = grids[0, HAZARD_CHANNELS:]
    assert player.sum() == 2, "one marked cell per decision frame"
    for channel in player:
        assert channel[16, 16] == 1.0


def test_player_channels_follow_the_player_and_clamp_at_the_edge():

    def cell_for(x, y):
        observation = torch.zeros(1, OBSERVATION_SIZE)
        for frame in range(2):
            observation[0, frame * FRAME_OBSERVATION_SIZE] = x / 127.0
            observation[0, frame * FRAME_OBSERVATION_SIZE + 1] = y / 127.0
        _, grids = DodgeHybridExtractor.policy_inputs(observation)
        marked = torch.nonzero(grids[0, HAZARD_CHANNELS])
        return tuple(marked[0].tolist())

    assert cell_for(0, 0) == (0, 0)
    assert cell_for(127, 127) == (GRID_SIZE - 1, GRID_SIZE - 1)
    # row is y, column is x -- a transposition here would be invisible on the
    # diagonal cases above.
    assert cell_for(100, 20) == (5, 25)


def test_hazard_channels_keep_their_order_and_meaning():
    """Old checkpoints migrate by copying into the leading channels."""

    observation = torch.zeros(1, OBSERVATION_SIZE)
    # One lethal hazard in the current frame's first cell.
    observation[0, FRAME_OBSERVATION_SIZE + 2] = CELL_TYPES / CELL_TYPES
    _, grids = DodgeHybridExtractor.policy_inputs(observation)

    assert HAZARD_CHANNELS == 16
    lethal_channel = grids[0, CELL_TYPES + CELL_TYPES - 1]
    assert lethal_channel[0, 0] == 1.0
