# -*- coding: utf-8 -*-
from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy
from chainer.functions.pooling import pooling_2d


class L1Penalty(function.Function):
    def __init__(self, lamda=0.1):
        self.lamda = lamda

    def forward(self, x):
        return x[0],

    def backward_cpu(self, x, gy):
        lamda = self.lamda / x[0].size
        gx = gy[0] + lamda * numpy.sign(x[0])
        return gx,

    def backward_gpu(self, x, gy):
        lamda = self.lamda / x[0].size
        gx = gy[0] + lamda * cuda.cupy.sign(x[0])
        return gx,

def l1_penalty(x, lamda=0.1):
    return L1Penalty(lamda)(x)




class AverageTemporalPooling2D(pooling_2d.Pooling2D):
    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 5
        )

    def forward(self, x):
        self.T = x[0].shape[2]
        y = x[0].mean(axis=(2))
        return y,

    def backward(self, x, gy):
        xp = cuda.get_array_module(x[0])
        gx = xp.tile(gy[0][:, :, xp.newaxis, :, :],
                     (1, 1, self.T, 1, 1))
        gx /= self.T
        return gx,


def average_temporal_pooling_2d(x, ksize=4, stride=None, pad=0, use_cudnn=True):
    """Spatial average pooling function.

    This function acts similarly to :class:`~functions.Convolution2D`, but
    it computes the average of input spatial patch for each channel
    without any parameter instead of computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or pair of ints): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent. If ``None`` is
            specified, then it uses same stride as the pooling window size.
        pad (int or pair of ints): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    .. note::

       This function currently does not support ``cover_all`` mode as
       :func:`max_pooling_2d`. Average pooling runs in non-cover-all mode.

    """
    return AverageTemporalPooling2D(ksize, stride, pad, False, use_cudnn)(x)