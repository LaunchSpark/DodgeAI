"""How the velocity-flow policies run fast, without changing what they compute.

Everything here is execution, not maths: which cuDNN kernels a convolution
uses, which memory layout it reads, and whether a forward pass is launched op
by op or replayed as one captured CUDA graph. tests/test_policy_parity.py holds
each of these to the outputs recorded before any of them existed.
"""

import os
import warnings

import torch
from torch.nn import functional as F

# Stable-Baselines3 seeds a CUDA run with cudnn.deterministic = True, which
# restricts every convolution to deterministic kernels. For the v2 dilated
# convolution that means a "grouped direct" backward-data kernel taking ~25ms
# of a ~70ms minibatch on a GTX 1070, where the default kernel takes ~2ms.
#
# It buys nothing here. The network's bilinear upsampling, grid_sample and
# adaptive pooling all backpropagate through atomic adds on CUDA, so its
# gradients already differ from one identical backward pass to the next with
# the flag on (measured: 13 of 25 parameter gradients, every repeat). Bit-exact
# resume is a CPU property, and on the CPU none of this applies.
#
# Set DODGE_DETERMINISTIC_CUDNN=1 to keep Stable-Baselines3's setting.
DETERMINISTIC_CUDNN = os.environ.get("DODGE_DETERMINISTIC_CUDNN", "") not in ("", "0")


def configure_cudnn():
    """Let cuDNN pick its fastest kernels, unless told to stay deterministic.

    Called whenever a velocity-flow policy is built, which is after
    Stable-Baselines3 has seeded the run (and set its own flags), both for a
    new model and for one loaded from a checkpoint.
    """
    if DETERMINISTIC_CUDNN or not torch.cuda.is_available():
        return
    torch.backends.cudnn.deterministic = False
    # Every shape here is fixed for a run -- n_envs for rollouts, the
    # minibatch for updates -- so autotuning is paid once per shape.
    torch.backends.cudnn.benchmark = True


# oneDNN runs these small convolutions markedly faster on channels-last input:
# roughly 7.5 -> 5.9ms a forward at batch 8 and 42 -> 27ms at batch 64 on this
# project's 4-core CPU, and the update's backward pass gains about as much.
# The layout changes the order oneDNN accumulates in, so the floats move by
# float32 rounding (~1e-6 relative) and no more. On a Pascal GPU, with no
# tensor cores, the same layout makes cuDNN slower, so it is CPU-only.
CPU_CHANNELS_LAST = True


def trunk_layout(grids):
    """The memory layout the convolutional trunk should read its input in."""
    if CPU_CHANNELS_LAST and grids.device.type == "cpu":
        return grids.contiguous(memory_format=torch.channels_last)
    return grids


# PyTorch's channels-first bilinear upsampling kernel on CUDA parallelises over
# output pixels only, looping over batch and channel inside each thread: 1024
# threads for a 32x32 output however large the batch. The channels-last kernel
# parallelises over every output element. Converting there and back is cheap
# against that -- roughly 12 -> 10.5ms for a batch-256 forward and 34 -> 33ms
# for a minibatch update on a GTX 1070. Same interpolation weights; the floats
# differ only by rounding, and CUDA's upsampling backward was never
# deterministic anyway (atomic adds).
CUDA_CHANNELS_LAST_UPSAMPLE = True


def upsample(features, size):
    """Bilinear upsampling to `size`, align_corners=False, as F.interpolate."""
    if CUDA_CHANNELS_LAST_UPSAMPLE and features.is_cuda:
        features = features.contiguous(memory_format=torch.channels_last)
        return F.interpolate(features, size=size, mode="bilinear",
                             align_corners=False).contiguous()
    return F.interpolate(features, size=size, mode="bilinear", align_corners=False)


# avg_pool2d's CUDA kernels are written for arbitrary windows and strides, and
# a 2x2 non-overlapping window is the case they are worst at: 5.2ms forward and
# backward for the v2 stem at minibatch 512 on a GTX 1070. Viewing each 2x2
# block as two trailing axes and averaging them is the same mean -- the four
# values are summed and divided by four either way, in a different order --
# and runs as a plain reduction, 3.1ms. On the CPU it is the other way round:
# oneDNN's pooling is several times faster than the reduction, channels-last
# especially, so the CPU keeps avg_pool2d.
CUDA_VIEW_POOL = True


def halve(features):
    """2x2 average pooling, stride 2, as F.avg_pool2d(features, 2)."""
    if CUDA_VIEW_POOL and features.is_cuda:
        return block_mean(features)
    return F.avg_pool2d(features, 2)


def block_mean(features):
    """F.avg_pool2d(features, 2) as a reduction over each 2x2 block."""
    n, c, h, w = features.shape
    return features.view(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


# A 3x3 convolution with dilation 2 and padding 2 only ever combines pixels of
# the same parity: output (2i+p, 2j+q) reads inputs (2i+p+2a, 2j+q+2b). So it
# is exactly four independent undilated 3x3 convolutions, padding 1, one per
# parity sub-grid x[..., p::2, q::2], with the same weight and bias -- and the
# zero padding matches too, since a sub-grid's out-of-range neighbour is the
# full grid's out-of-range neighbour. Stacked into the batch they are one
# ordinary convolution, which cuDNN has far better kernels for than dilated
# ones: 10.9 -> 7.5ms forward and backward at minibatch 512 on a GTX 1070,
# shuffles included. Each output sums the same nine products per channel, so
# only the accumulation order changes. On the CPU oneDNN's dilated kernel is
# the faster of the two, and the CPU keeps it.
#
# One consequence of handing cuDNN an ordinary convolution: it may autotune to
# a Winograd-style kernel, which no dilated convolution can have, and those
# return ~1e-9 noise where the exact answer is 0. That is rounding like any
# other, except on LeakyReLU's kink: a freshly initialised policy has zero
# biases and so exact-zero pre-activations wherever the arena is empty, and
# which side of the kink they land on moves its first gradient by percent.
# The old route's undilated layers were always exposed to the same thing;
# tests/test_velocity_trunk.py explains how its gradient tests allow for it.
#
# Under DODGE_DETERMINISTIC_CUDNN=1 it is not a matter of luck: without
# autotuning, cuDNN's heuristics give the split exactly such a kernel every
# time, and the fresh policy's first gradient then misses the recorded one in
# tests/test_policy_parity.py by ~6%. Whoever sets that variable is asking for
# Stable-Baselines3's own behaviour, so it keeps the dilated convolution.
CUDA_PARITY_DILATED_CONV = not DETERMINISTIC_CUDNN


def dilated_conv(conv, features):
    """conv(features), for a 3x3 nn.Conv2d with dilation 2 and padding 2."""
    if CUDA_PARITY_DILATED_CONV and features.is_cuda:
        return parity_conv(conv, features)
    return F.conv2d(features, conv.weight, conv.bias, padding=2, dilation=2)


def parity_conv(conv, features):
    """The same convolution, as one undilated one over the parity sub-grids.

    Needs an even height and width, so the four sub-grids are the same size.
    """
    n, c, h, w = features.shape
    planes = (features.view(n, c, h // 2, 2, w // 2, 2)
              .permute(0, 3, 5, 1, 2, 4).reshape(n * 4, c, h // 2, w // 2))
    out = F.conv2d(planes, conv.weight, conv.bias, padding=1)
    return (out.view(n, 2, 2, -1, h // 2, w // 2)
            .permute(0, 3, 4, 1, 5, 2).reshape(n, -1, h, w))


# At rollout batch sizes a CUDA forward is ~150 kernel launches whose launch
# overhead, not the GPU, is nearly all of its cost: 4.8ms for a batch of 8,
# with under 1ms of that on the GPU. Captured as a CUDA graph it is one launch.
CUDA_GRAPHS = os.environ.get("DODGE_CUDA_GRAPHS", "1") not in ("", "0")
# Calls at one batch size before it is captured, so a size seen once (the
# terminal observation of an episode, a stray predict) never pays for one.
GRAPH_AFTER_CALLS = 3
# Each graph pins its own activations; rollouts need one.
MAX_GRAPHS = 4


class CapturedForward:
    """One no-grad forward pass at a fixed input shape, replayed as a graph.

    The graph reads parameters at the addresses it was captured with. Adam and
    load_state_dict both update in place, so the graph follows training; the
    owner checks the addresses before each replay in case something replaced
    a tensor outright.
    """

    def __init__(self, function, example):
        device = example.device
        self.static_input = example.clone()
        # Warm up on a side stream, as capture requires: cuDNN autotuning and
        # lazy allocations must happen before capture, not inside it.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                function(self.static_input)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_outputs = function(self.static_input)

    def __call__(self, inputs):
        self.static_input.copy_(inputs)
        self.graph.replay()
        return self.static_outputs


class GraphCache:
    """Captured forwards keyed by input shape, for one policy."""

    def __init__(self):
        self.graphs = {}
        self.calls = {}
        self.signature = None
        self.disabled = False

    def clear(self):
        self.graphs.clear()
        self.calls.clear()
        self.signature = None

    def lookup(self, function, inputs, signature):
        """A captured forward for these inputs, or None to run eagerly."""
        if self.disabled:
            return None
        if signature != self.signature:
            # A parameter or buffer was replaced rather than updated in place
            # (a device move, say): every graph is reading stale memory.
            self.clear()
            self.signature = signature
        key = (tuple(inputs.shape), inputs.dtype, inputs.device)
        graph = self.graphs.get(key)
        if graph is not None:
            return graph
        seen = self.calls.get(key, 0) + 1
        self.calls[key] = seen
        if seen < GRAPH_AFTER_CALLS or len(self.graphs) >= MAX_GRAPHS:
            return None
        try:
            graph = CapturedForward(function, inputs)
        except Exception as error:  # capture is best-effort; eager is correct
            self.disabled = True
            self.clear()
            warnings.warn(f"CUDA graph capture failed, running eagerly: {error}",
                          RuntimeWarning)
            return None
        self.graphs[key] = graph
        return graph
