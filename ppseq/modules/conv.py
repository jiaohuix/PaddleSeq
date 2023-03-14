import math
import paddle
import paddle.nn.initializer as init

def conv2d(in_channels,out_channels,kernel_size,**kwargs):
    in_fan = in_channels * kernel_size * kernel_size
    m = paddle.nn.Conv2D(in_channels, out_channels, kernel_size, **kwargs,
                                 weight_attr=init.Uniform(-1 / math.sqrt(in_fan), 1 / math.sqrt(in_fan)),
                                 bias_attr=init.Uniform(-1 / math.sqrt(in_fan), 1 / math.sqrt(in_fan)))
    return m


