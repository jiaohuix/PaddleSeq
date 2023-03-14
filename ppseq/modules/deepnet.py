import paddle.nn as nn
import paddle.nn.initializer as I
from fastcore.all import patch_to, partial # 1.0
from paddle.nn.layer.transformer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
import functools
import paddle.nn.initializer as I
xavier_uniform_=I.XavierUniform()

@patch_to(nn.Layer)
def apply(self, fn, name=""):
    for n, layer in self.named_children():
        nnmame = n if name == "" else name + "." + n
        layer.apply(fn, nnmame)

    fn(self, name)
    return self


def xavier_uniform_with_gain(tensor,gain):
    xavier_uniform_ = I.XavierUniform()
    xavier_uniform_._compute_fans = decorator(
        xavier_uniform_._compute_fans, gain=gain
    )
    xavier_uniform_(tensor)

def decorator(func, gain=1):
    @functools.wraps(func)
    def wrappper(*args, **kwargs):
        fan_in, fan_out = func(*args, **kwargs)
        return fan_in / (gain ** 2), fan_out / (gain ** 2)

    return wrappper


def xavier_normal_fn(weight,gain=1):
    ''' with torch init '''
    try:
        import torch
        import torch.nn as tnn
        w = torch.from_numpy(weight.numpy())
        w = tnn.init.xavier_normal_(w,gain=gain)
        weight.set_value(w.numpy())
    except ImportError as err:
        xavier_normal_gain = I.XavierNormal()
        xavier_normal_gain._compute_fans = decorator(
            xavier_normal_gain._compute_fans, gain=gain
        )
        xavier_normal_gain(weight)


def deepnorm_init(m, n, N=6, M=6):
    ''' N:encoder layers, M: decoder layers '''
    if "encoder" in n:
        alpha = 0.81 * ((N ** 4) * M) ** (1 / 16)
        beta = 0.87 * ((N ** 4) * M) ** -(1 / 16)
    elif "decoder" in n:
        alpha = (3 * M) ** (1 / 4)
        beta = (12 * M) ** -(1 / 4)
    else:
        return

    if isinstance(m, nn.Linear):
        if any(x in n for x in ["linear1", "linear2", "v_proj", "out_proj"]):
            xavier_normal_fn(m.weight,gain=beta)
        elif any(x in n for x in ["q_proj", "k_proj"]):
            xavier_normal_fn(m.weight,gain=1)
    if isinstance(m, TransformerEncoderLayer) and "encoder" in n:
        setattr(m, "alpha", alpha)
    elif isinstance(m, TransformerDecoderLayer) and "decoder" in n:
        setattr(m, "alpha", alpha)
