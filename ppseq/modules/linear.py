# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import paddle
import math
import paddle.nn as nn
import paddle.nn.initializer as init
import paddle.nn.functional as F

def Linear(in_features,out_features,bias=True):
    m = nn.Linear(in_features,out_features,
                    weight_attr=init.Uniform(-1 / math.sqrt(in_features), 1 / math.sqrt(in_features)),
                    bias_attr=init.Uniform(-1/math.sqrt(in_features), 1/math.sqrt(in_features)) if bias else bias)
                    # bias_attr=init.Constant(value=0.0) if bias else bias)
    return m


class Mlp(nn.Layer):
    def __init__(self,d_model,dim_feedforward,drop=0.,activation="relu"):
        super(Mlp, self).__init__()
        self.act=getattr(F,activation)
        self.dropout=nn.Dropout(p=drop)
        self.linear1=Linear(d_model,dim_feedforward)
        self.linear2=Linear(dim_feedforward,d_model)

    def forward(self,x):
        x=self.act(self.linear1(x))
        x=self.linear2(self.dropout(x))
        return x
