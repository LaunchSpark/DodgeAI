"""The velocity-flow-v2 policy computes what it computed before the speed work.

Compared against tests/data/policy_parity.npz, recorded by tests/policy_parity.py
from the unoptimised policy. Two things are held:

* Sampled actions are identical. Same observations, same torch seed, same
  actions -- on the CPU always, and on CUDA against the CUDA recording.
* Every float PPO reads is within float32 rounding of the recording, measured
  against the largest magnitude in each array (see the tolerances below).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from tests import policy_parity

# Each tolerance is (rel, abs): max|current - recorded| <= rel * max|recorded|
# + abs, per recorded array. float32 rounding in a reordered sum is ~1e-7 of
# its operands, and the trunk sums hundreds of them.
#
# Forward outputs -- features, logits, log-probs, values, entropy -- on any
# device. The logits set the bar: _FieldLogits divides the nine path dangers by
# their spread, which is small next to the dangers themselves, so it magnifies
# rounding. The unchanged code's own CPU and CUDA logits differ by 1.0e-5 of
# their largest magnitude; after the speed work the CPU's differ from the
# recording by 1.4e-5, and everything else by under 1e-6.
FORWARD_TOLERANCE = (3e-5, 1e-6)
# Per-tensor gradient checksums of a PPO-shaped loss.
GRADIENT_TOLERANCE = (1e-4, 1e-9)
# Adam's first steps are close to sign(gradient), so a gradient element near
# zero that moves by rounding can flip its step outright. CUDA's backward is
# itself nondeterministic here (atomic adds in upsampling and grid_sample):
# the unchanged code, run twice, differed by ~5e-5 on this checksum.
ADAM_TOLERANCE = {"cpu": (1e-4, 1e-9), "cuda": (5e-4, 1e-9)}
# CUDA against the CPU recording: different kernels, same maths.
CROSS_DEVICE_TOLERANCE = FORWARD_TOLERANCE

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def assert_close(current, recorded, tolerance, name):
    rel, floor = tolerance
    current, recorded = np.asarray(current), np.asarray(recorded)
    assert current.shape == recorded.shape, name
    worst = float(np.abs(current - recorded).max())
    bound = rel * float(np.abs(recorded).max()) + floor
    assert worst <= bound, f"{name}: off by {worst:.3g}, allowed {bound:.3g}"


@pytest.fixture(scope="module")
def recording():
    if not policy_parity.FIXTURE.exists():
        pytest.skip("no parity recording")
    with np.load(policy_parity.FIXTURE) as data:
        return dict(data)


@pytest.fixture(scope="module", params=DEVICES)
def current(request, recording):
    if not policy_parity.CHECKPOINT.exists():
        pytest.skip("velocity-flow-v2-latest.zip is not present")
    device = request.param
    obs = recording["observations"]
    return device, {name: policy_parity.outputs(policy, obs)
                    for name, policy in policy_parity.load_policies(device).items()}


def test_parity_recording_has_both_policies(recording):
    assert recording["observations"].shape[0] == 64
    for name in ("latest", "fresh"):
        assert f"cpu/{name}/logits" in recording


@pytest.mark.parametrize("name", ["latest", "fresh"])
def test_sampled_actions_are_identical(current, recording, name):
    device, results = current
    key = f"{device}/{name}"
    if f"{key}/logits" not in recording:
        pytest.skip(f"nothing recorded on {device}")
    actions = [k for k in results[name] if k.endswith("_actions")]
    assert actions
    for k in actions + ["deterministic"]:
        np.testing.assert_array_equal(results[name][k], recording[f"{key}/{k}"],
                                      err_msg=k)


@pytest.mark.parametrize("name", ["latest", "fresh"])
def test_forward_and_update_outputs_match(current, recording, name):
    device, results = current
    key = f"{device}/{name}"
    if f"{key}/logits" not in recording:
        pytest.skip(f"nothing recorded on {device}")
    for k, value in results[name].items():
        if k.endswith("_actions") or k == "deterministic":
            continue
        if k.startswith("adam_"):
            tolerance = ADAM_TOLERANCE[device]
        elif k.startswith("grad_"):
            tolerance = GRADIENT_TOLERANCE
        else:
            tolerance = FORWARD_TOLERANCE
        assert_close(value, recording[f"{key}/{k}"], tolerance, k)


def test_cuda_agrees_with_cpu(recording):
    """Cross-device: the same maths, so floats agree to float32 rounding."""
    if "cuda/latest/logits" not in recording:
        pytest.skip("nothing recorded on cuda")
    for name in ("latest", "fresh"):
        for k in ("features", "logits", "log_softmax", "values", "entropy"):
            assert_close(recording[f"cuda/{name}/{k}"], recording[f"cpu/{name}/{k}"],
                         CROSS_DEVICE_TOLERANCE, f"{name}/{k}")


def test_real_observations_still_match_the_recording(recording):
    """The live env still produces the observations the recording was made on."""
    np.testing.assert_array_equal(policy_parity.real_observations(),
                                  recording["observations"][:48])


# --- the fast paths against Stable-Baselines3's own, live -----------------

def _live(device):
    if not policy_parity.CHECKPOINT.exists():
        pytest.skip("velocity-flow-v2-latest.zip is not present")
    policy = policy_parity.load_policies(device)["latest"]
    policy.set_training_mode(False)
    obs = torch.as_tensor(policy_parity.synthetic_observations(), device=device)
    return policy, obs


@pytest.mark.parametrize("device", DEVICES)
def test_forward_is_sb3_forward_bit_for_bit(device):
    from stable_baselines3.common.policies import ActorCriticPolicy
    from dodge import speed

    policy, obs = _live(device)
    previous, speed.CUDA_GRAPHS = speed.CUDA_GRAPHS, False
    try:
        for deterministic in (False, True):
            with torch.no_grad():
                torch.manual_seed(3)
                ours = policy(obs, deterministic=deterministic)
                torch.manual_seed(3)
                theirs = ActorCriticPolicy.forward(policy, obs, deterministic)
            for a, b in zip(ours, theirs):
                assert torch.equal(a, b)
    finally:
        speed.CUDA_GRAPHS = previous


@pytest.mark.parametrize("device", DEVICES)
def test_evaluate_actions_is_sb3_bit_for_bit_with_gradients(device):
    from stable_baselines3.common.policies import ActorCriticPolicy

    policy, obs = _live(device)
    if device == "cuda":
        pytest.skip("CUDA backward is nondeterministic run to run")
    policy.set_training_mode(True)
    actions = torch.arange(len(obs), device=device) % 9
    results = []
    for evaluate in (policy.evaluate_actions,
                     lambda o, a: ActorCriticPolicy.evaluate_actions(policy, o, a)):
        policy.zero_grad()
        values, log_prob, entropy = evaluate(obs, actions)
        (log_prob.sum() + values.sum() + entropy.sum()).backward()
        results.append([values, log_prob, entropy]
                       + [p.grad.clone() for p in policy.parameters()])
    for a, b in zip(*results):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_replays_the_eager_forward_and_follows_training():
    from dodge import speed

    policy, obs = _live("cuda")
    batch = obs[:8]

    def run(graphs):
        previous, speed.CUDA_GRAPHS = speed.CUDA_GRAPHS, graphs
        try:
            with torch.no_grad():
                torch.manual_seed(5)
                return policy(batch)
        finally:
            speed.CUDA_GRAPHS = previous

    for _ in range(speed.GRAPH_AFTER_CALLS + 1):
        run(True)
    assert policy._graphs.graphs, "no graph was captured"
    for a, b in zip(run(True), run(False)):
        assert torch.equal(a, b)

    # An optimizer step updates parameters in place; the graph must see it.
    policy.set_training_mode(True)
    values, log_prob, entropy = policy.evaluate_actions(obs, torch.zeros(len(obs), device="cuda"))
    policy.optimizer.zero_grad()
    (log_prob.mean() + values.mean()).backward()
    policy.optimizer.step()
    policy.set_training_mode(False)
    for a, b in zip(run(True), run(False)):
        assert torch.equal(a, b)


def test_policy_still_deep_copies_after_capturing():
    import copy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, obs = _live(device)
    with torch.no_grad():
        for _ in range(4):
            policy(obs[:8])
    clone = copy.deepcopy(policy)
    with torch.no_grad():
        torch.manual_seed(1)
        a = clone(obs[:8])
        torch.manual_seed(1)
        b = policy(obs[:8])
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_seeded_cuda_model_does_not_force_deterministic_cudnn():
    """SB3 turns cudnn.deterministic on when it seeds; building v2 turns it off."""
    from stable_baselines3 import PPO
    from dodge import speed

    if speed.DETERMINISTIC_CUDNN:
        pytest.skip("DODGE_DETERMINISTIC_CUDNN is set")
    PPO.load(policy_parity.CHECKPOINT, device="cuda", seed=0)
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark


def test_cpu_v2_training_is_bit_reproducible():
    """Exact resume rests on this: the CPU fast paths must stay deterministic."""
    from stable_baselines3 import PPO
    from dodge.env import DodgeVecEnv
    from dodge.policies import policy_class, policy_kwargs

    def train():
        env = DodgeVecEnv(n_envs=2, seed=3, action_repeat=4)
        try:
            model = PPO(policy_class("velocity-flow-v2"), env, n_steps=32,
                        batch_size=32, n_epochs=2, seed=0, device="cpu",
                        policy_kwargs=policy_kwargs("velocity-flow-v2"))
            model.learn(128)
        finally:
            env.close()
        return model.policy.state_dict()

    first, second = train(), train()
    assert all(torch.equal(first[k], second[k]) for k in first)
