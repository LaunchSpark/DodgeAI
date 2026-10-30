"""The velocity-flow-v2 policy computes what it computed before the speed work.

Compared against tests/data/policy_parity.npz, recorded by tests/policy_parity.py
from the unoptimised policy. Two things are held:

* Sampled actions are identical. Same observations, same torch seed, same
  actions -- on the CPU always, and on CUDA against the CUDA recording.
* Every float PPO reads is within float32 rounding of the recording:
  FLOAT_TOLERANCE for forward outputs, a relative GRADIENT_TOLERANCE for the
  gradient and Adam checksums, which sum over whole parameter tensors.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from tests import policy_parity

# Forward outputs: logits, log-probs, values, entropy, features.
FLOAT_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}
# Per-tensor sums of gradients and Adam steps, relative to their own scale.
GRADIENT_TOLERANCE = {"rtol": 1e-4, "atol": 1e-7}
CUDA_TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


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
def test_forward_outputs_match(current, recording, name):
    device, results = current
    key = f"{device}/{name}"
    if f"{key}/logits" not in recording:
        pytest.skip(f"nothing recorded on {device}")
    tolerance = FLOAT_TOLERANCE if device == "cpu" else CUDA_TOLERANCE
    for k, value in results[name].items():
        if k.endswith("_actions") or k == "deterministic":
            continue
        reference = recording[f"{key}/{k}"]
        if k.startswith(("grad_", "adam_")):
            scale = max(float(np.abs(reference).max()), 1e-12)
            np.testing.assert_allclose(value / scale, reference / scale,
                                       rtol=0, atol=GRADIENT_TOLERANCE["rtol"],
                                       err_msg=k)
        else:
            np.testing.assert_allclose(value, reference, err_msg=k, **tolerance)


def test_cuda_agrees_with_cpu(recording):
    """Cross-device: the same maths, so floats agree to CUDA_TOLERANCE."""
    if "cuda/latest/logits" not in recording:
        pytest.skip("nothing recorded on cuda")
    for name in ("latest", "fresh"):
        for k in ("features", "logits", "log_softmax", "values", "entropy"):
            np.testing.assert_allclose(recording[f"cuda/{name}/{k}"],
                                       recording[f"cpu/{name}/{k}"],
                                       err_msg=f"{name}/{k}", **CUDA_TOLERANCE)


def test_real_observations_still_match_the_recording(recording):
    """The live env still produces the observations the recording was made on."""
    np.testing.assert_array_equal(policy_parity.real_observations(),
                                  recording["observations"][:48])
