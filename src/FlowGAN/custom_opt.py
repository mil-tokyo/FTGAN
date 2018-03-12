# -*- coding: utf-8 -*-
from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy

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
