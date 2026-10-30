"""A frozen copy of VelocityFlowV2Extractor's forward pass, as it was computed
before the half-resolution trunk stopped materialising its 32x32 output.

It reads the live extractor's parameters but none of its methods, so whatever
dodge/velocity.py does to route the same maths more cheaply, this is the route
it is held against: the dilated convolution run dilated, the trunk upsampled
to 32x32, pooled, projected by the 1x1 heads, and read with grid_sample.

Do not "fix" or speed this up. Its only job is to stay what it was.
"""

import torch
from torch import nn
from torch.nn import functional as F

from dodge.observe import ARENA_SIZE, CELL_SIZE, GRID_SIZE
from dodge.policies import DodgeHybridExtractor, _sense_features
from dodge.velocity import (CONTEXT_GRID, N_ACTIONS, SAMPLE_FRAMES,
                            VALUE_SUMMARY_GRID)


def _upsample(features, size):
    return F.interpolate(features, size=size, mode="bilinear", align_corners=False)


def _run(sequential, x):
    """An nn.Sequential, with every convolution run as a plain nn.Conv2d."""
    for layer in sequential:
        x = nn.Conv2d.forward(layer, x) if isinstance(layer, nn.Conv2d) else layer(x)
    return x


def trunk(extractor, grids):
    """The 32x32 trunk: stem, 2x average pool, dilated body, bilinear back up.

    Or, for v1 and for v2 built with half_resolution=False, the plain
    full-resolution stack.
    """
    net = extractor.field_net
    if isinstance(net, nn.Sequential):
        return _run(net, grids)
    stem = _run(net.stem, grids)
    body = _run(net.body, F.avg_pool2d(stem, 2))
    return _upsample(body, stem.shape[-2:])


def field_from_trunk(extractor, trunk_):
    context = _run(extractor.context_net, F.adaptive_avg_pool2d(trunk_, CONTEXT_GRID))
    context = _upsample(context, trunk_.shape[-2:])
    return nn.Conv2d.forward(extractor.field_head, torch.cat((trunk_, context), dim=1))


def field(extractor, observations):
    _, grids = DodgeHybridExtractor.policy_inputs(observations)
    return field_from_trunk(extractor, trunk(extractor, grids))


def forward(extractor, observations):
    """The extractor's forward pass, as it was: v1, or v2 at either resolution."""
    coordinates, grids = DodgeHybridExtractor.policy_inputs(observations)
    trunk_ = trunk(extractor, grids)
    field_ = field_from_trunk(extractor, trunk_)

    position = coordinates[:, :2] * (ARENA_SIZE - 1)
    points = position[:, None, None, :] + extractor.offsets[None]
    normalised = points / (GRID_SIZE * CELL_SIZE / 2.0) - 1.0
    batch = observations.shape[0]
    sampled = F.grid_sample(
        field_, normalised.reshape(batch, N_ACTIONS, len(SAMPLE_FRAMES), 2),
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    paired = sampled.permute(0, 2, 1, 3).diagonal(dim1=2, dim2=3)
    danger = (paired * extractor.sample_weights).sum(dim=-1)

    senses = _sense_features(coordinates, grids)
    pooled = trunk_.mean(dim=(2, 3))
    if not hasattr(extractor, "value_pool"):              # v1
        value = extractor.value_net(torch.cat((pooled, coordinates, senses), dim=1))
        return torch.cat((-danger, value), dim=1)
    summary = F.adaptive_avg_pool2d(
        nn.Conv2d.forward(extractor.value_pool, trunk_), VALUE_SUMMARY_GRID).flatten(1)
    value = extractor.value_net(torch.cat((pooled, summary, coordinates, senses), dim=1))
    return torch.cat((-danger, value), dim=1)
