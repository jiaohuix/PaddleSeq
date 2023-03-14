import paddle.nn as nn
from ppseq.modules.torch_utils import normal_fn_,make_pad_zero
import paddle.nn.initializer as init

def Embedding(num_embeddings, embedding_dim, padding_idx=1):
    m = nn.Embedding(num_embeddings,embedding_dim, weight_attr=init.Normal(0.0, 1.0))
    # normalize
    normal_fn_(m.weight,rand_norm=True,mean=0,std=embedding_dim ** -0.5)
    # remove pad
    make_pad_zero(m.weight,padding_idx)
    return m