# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from paddle.nn import MultiHeadAttention
from ppseq.modules import Linear
from .initializer import xavier_uniform_,xavier_normal_

class MultiHeadAttentionWithInit(MultiHeadAttention):
    def __init__(self,use_deepnorm=False,*args,**kwargs):
        super(MultiHeadAttentionWithInit,self).__init__(*args,**kwargs)
        self.k_proj = Linear(self.kdim, self.embed_dim)
        self.v_proj = Linear(self.vdim, self.embed_dim)
        self.q_proj = Linear(self.embed_dim, self.embed_dim)
        self.out_proj = Linear(self.embed_dim, self.embed_dim)

        self.qkv_same_dim = self.kdim == self.embed_dim and self.vdim == self.embed_dim

        if not use_deepnorm:
            self.reset_paramaters()

    def reset_paramaters(self):
        if self.qkv_same_dim: # <-
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            xavier_uniform_(self.k_proj.weight)
            xavier_uniform_(self.v_proj.weight)
            xavier_uniform_(self.q_proj.weight)

        xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            from paddle.nn.initializer import Constant
            zero_ = Constant(value=0.0)
            zero_(self.out_proj.bias)


