# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
LayerDrop as described in https://arxiv.org/abs/1909.11556.
"""

import paddle
import paddle.nn as nn


class LayerDropList(nn.LayerList):
    """
    A LayerDrop implementation based on :class:`paddle.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, layers=None):
        super().__init__(layers)
        assert p<=1 and p>=0,"p value err."
        self.p = p

    def __iter__(self):
        dropout_probs = paddle.uniform(shape=[len(self)],min=0.,max=1.)
        for i, m in enumerate(super().__iter__()):
            # when training, drop layer when probs[i]<=p
            if not self.training or (dropout_probs[i] > self.p):
                yield m
