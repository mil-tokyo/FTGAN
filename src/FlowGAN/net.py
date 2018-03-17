#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

from chainer import Variable
from custom_opt import l1_penalty

class Generator(chainer.Chain):
    def __init__(self, video_len = 32):
        w = chainer.initializers.Normal(0.02)
        self.video_len = video_len
        super(Generator, self).__init__(
            l0=L.Linear(100, 4*4*512*(self.video_len//16), initialW=w),
            dc1=L.DeconvolutionND(3, 512, 256, 4, 2, 1, initialW=w),
            dc2=L.DeconvolutionND(3, 256, 128, 4, 2, 1, initialW=w),
            dc3=L.DeconvolutionND(3, 128, 64, 4, 2, 1, initialW=w),
            dc_fore=L.DeconvolutionND(3, 64, 2, 4, 2, 1, initialW=w),
            dc_mask=L.DeconvolutionND(3, 64, 1, 4, 2, 1, initialW=w),

            bn0=L.BatchNormalization(4*4*512*(self.video_len/16)),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def make_hidden(self, batchsize):
        return numpy.random.normal(0, 1, (batchsize, 100, 1, 1))\
            .astype(numpy.float32)

    def __call__(self, z):
        h = F.reshape(F.relu(self.bn0(self.l0(z))),
                      (z.data.shape[0], 512, (self.video_len//16), 4, 4))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h_fore = F.tanh(self.dc_fore(h))

        h_mask = F.sigmoid(self.dc_mask(h))
        h_mask = l1_penalty(h_mask)
        h_mask_rep = F.tile(h_mask, (1, 2, 1, 1, 1))

        x = h_mask_rep * h_fore
        if chainer.config.train:
            return x
        else:
            return x, h_fore, h_mask

class Discriminator(chainer.Chain):

    def __init__(self):
        w = chainer.initializers.Normal(0.02)
        super(Discriminator, self).__init__(
            c0=L.ConvolutionND(3, 2, 64, 4, 2, 1, initialW=w),
            c1=L.ConvolutionND(3, 64, 128, 4, 2, 1, initialW=w),
            c2=L.ConvolutionND(3, 128, 256, 4, 2, 1, initialW=w),
            c3=L.ConvolutionND(3, 256, 512, 4, 2, 1, initialW=w),
            l4=L.Linear(None, 1, initialW=w),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        return self.l4(h)

class GAN_Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.generator, self.discriminator = kwargs.pop('models')
        super(GAN_Updater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = y_fake.data.shape[0]
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = y_fake.data.shape[0]
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self._iterators['main'].next()
        flow_real = Variable(self.converter(batch, self.device))

        gen, dis = self.generator, self.discriminator
        xp = chainer.cuda.get_array_module(flow_real.data)
        y_real = dis(flow_real)
        batchsize = flow_real.data.shape[0]

        ### z or image
        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        flow_fake = gen(z)

        y_fake = dis(flow_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)
