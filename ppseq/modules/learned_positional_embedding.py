# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import paddle.nn as nn
from paddlenlp.transformers import PositionalEmbedding
from ppseq.modules.torch_utils import make_pad_zero
import paddle.nn.initializer as I


class PositionalEmbeddingLeanable(PositionalEmbedding):
    def __init__(self,
                 pad_idx=1,
                 learnable=False,
                 learned_sinusoidal=False,
                 *args,**kwargs):
        super(PositionalEmbeddingLeanable,self).__init__(*args,**kwargs)
        self.pad_idx= pad_idx
        self.learnable = learnable
        self.learned_sinusoidal = learned_sinusoidal
        emb_dim, max_length = kwargs.get("emb_dim"), kwargs.get("max_length")

        if learnable:
            self.pos_encoder = nn.Embedding(num_embeddings=emb_dim,embedding_dim=max_length)
            nromal_ = I.Normal(mean=0, std=emb_dim ** -0.5)
            nromal_(self.pos_encoder.weight)

        make_pad_zero(self.pos_encoder.weight,pad_idx)
        if not self.learnable and not self.learned_sinusoidal:
            self.pos_encoder.weight.stop_gradient = True

    def forward(self, pos):
        pos_emb = self.pos_encoder(pos)
        if not self.learnable and not self.learned_sinusoidal:
            pos_emb.stop_gradient = True
        return pos_emb

