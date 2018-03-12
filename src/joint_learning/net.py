#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

import six
import math

from chainer import Variable
from custom_opt import l1_penalty, average_temporal_pooling_2d


class CBR(chainer.Chain):
    def AddNoise(self, h):
        xp = cuda.get_array_module(h.data)
        if chainer.config.train:
            return h + self.sigma * xp.random.randn(*h.data.shape)
        else:
            return h

    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.leaky_relu, add_noise=False, sigma=0.2):
        self.bn = bn
        self.activation = activation
        self.add_noise = add_noise
        self.sigma = sigma
        self.iteration = 0
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample == 'down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        elif sample == 'up':
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        elif sample == 'same':
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x):
        h = self.c(x)

        if self.bn:
            h = self.batchnorm(h)
        if self.add_noise:
            h = self.AddNoise(h)
            if chainer.config.train:
                self.iteration += 1
                if self.iteration % 5000 == 0 and self.iteration != 0:
                    self.sigma *= 0.5
        if not self.activation is None:
            h = self.activation(h)
        return h

class CBR3D(chainer.Chain):
    def AddNoise(self, h):
        xp = cuda.get_array_module(h.data)
        if chainer.config.train:
            return h + self.sigma * xp.random.randn(*h.data.shape)
        else:
            return h

    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.leaky_relu, add_noise=False, sigma=0.2):
        self.bn = bn
        self.activation = activation
        self.add_noise = add_noise
        self.sigma = sigma
        self.iteration = 0
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample == 'down':
            layers['c'] = L.ConvolutionND(3, ch0, ch1, 4, 2, 1, initialW=w)
        elif sample == 'up':
            layers['c'] = L.DeconvolutionND(3, ch0, ch1, 4, 2, 1, initialW=w)
        elif sample == 'same':
            layers['c'] = L.ConvolutionND(3, ch0, ch1, 3, 1, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR3D, self).__init__(**layers)

    def __call__(self, x):
        h = self.c(x)

        if self.bn:
            h = self.batchnorm(h)
        if self.add_noise:
            h = self.AddNoise(h)
            if chainer.config.train:
                self.iteration += 1
                if self.iteration % 5000 == 0 and self.iteration != 0:
                    self.sigma *= 0.5
        if not self.activation is None:
            h = self.activation(h)
        return h

### flow GAN
class FlowGenerator(chainer.Chain):
    def __init__(self, video_len=32):
        w = chainer.initializers.Normal(0.02)
        self.video_len = video_len
        super(FlowGenerator, self).__init__(
            l0=L.Linear(100, 4 * 4 * 512 * (self.video_len // 16), initialW=w),
            dc1=L.DeconvolutionND(3, 512, 256, 4, 2, 1, initialW=w),
            dc2=L.DeconvolutionND(3, 256, 128, 4, 2, 1, initialW=w),
            dc3=L.DeconvolutionND(3, 128, 64, 4, 2, 1, initialW=w),
            dc_fore=L.DeconvolutionND(3, 64, 2, 4, 2, 1, initialW=w),
            dc_mask=L.DeconvolutionND(3, 64, 1, 4, 2, 1, initialW=w),

            bn0=L.BatchNormalization(4 * 4 * 512 * (self.video_len / 16)),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def make_hidden(self, batchsize):
        return numpy.random.normal(0, 1, (batchsize, 100, 1, 1)) \
            .astype(numpy.float32)

    def __call__(self, z):
        h = F.reshape(F.relu(self.bn0(self.l0(z))),
                      (z.data.shape[0], 512, (self.video_len // 16), 4, 4))
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

class FlowDiscriminator(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.Normal(0.02)
        super(FlowDiscriminator, self).__init__(
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

### wo back dis
class Generator(chainer.Chain):
    def __init__(self, dimz, gf_dim=512, lamda=0.1):
        self.dimz = dimz
        self.gf_dim = gf_dim
        self.lamda = lamda
        w = chainer.initializers.Normal(0.02)
        super(Generator, self).__init__(
            ### fore img generator
            l_f0=L.Linear(dimz, 4 * 4 * gf_dim // 2 * 2, initialW=w),
            bn_f0=L.BatchNormalization(4 * 4 * gf_dim // 2 * 2),
            dc_f1=CBR3D(gf_dim // 2, gf_dim // 4, bn=True, sample='up', activation=F.relu),
            dc_f2=CBR3D(gf_dim // 4, gf_dim // 8, bn=True, sample='up', activation=F.relu),

            ### back img generator
            l_b0=L.Linear(self.dimz, 4 * 4 * gf_dim, initialW=w),
            bn_b0=L.BatchNormalization(4 * 4 * gf_dim),
            dc_b1=CBR(None, gf_dim // 2, bn=True, sample='up', activation=F.relu),
            dc_b2=CBR(None, gf_dim // 4, bn=True, sample='up', activation=F.relu),
            dc_b3=CBR(None, gf_dim // 8, bn=True, sample='up', activation=F.relu),
            dc_b4=L.Deconvolution2D(None, 3, 4, 2, 1, initialW=w),

            ### flow colorizer w U-net
            c_m1=CBR3D(2, gf_dim // 16, bn=False, sample='down', activation=F.leaky_relu),
            c_m2=CBR3D(gf_dim // 16, gf_dim // 8, bn=True, sample='down', activation=F.leaky_relu),
            c_m3=CBR3D(gf_dim // 4, gf_dim // 4, bn=True, sample='same', activation=F.leaky_relu),
            c_m4=CBR3D(gf_dim // 4, gf_dim // 2, bn=True, sample='down', activation=F.leaky_relu),
            c_m5=CBR3D(gf_dim // 2, gf_dim, bn=True, sample='down', activation=F.leaky_relu),

            dc_m1=CBR3D(gf_dim, gf_dim // 2, bn=True, sample='up', activation=F.relu),
            dc_m2=CBR3D(gf_dim, gf_dim // 4, bn=True, sample='up', activation=F.relu),
            dc_m3=CBR3D(gf_dim // 2, gf_dim // 8, bn=True, sample='up', activation=F.relu),
            dc_mask=L.DeconvolutionND(3, gf_dim // 8, 1, 4, 2, 1, initialW=w),
            dc_m4=CBR3D(gf_dim // 16 * 3, gf_dim // 16, bn=True, sample='up', activation=F.relu),
            dc_m5=L.ConvolutionND(3, gf_dim // 16, 3, 3, 1, 1, initialW=w),
        )

    def make_hidden(self, batchsize):
        return numpy.random.normal(0, 1, (batchsize, self.dimz, 1, 1)) \
            .astype(numpy.float32)

    def __call__(self, z, flow):
        B, CH, T, Y, X = flow.shape

        ### back img generation
        h = F.reshape(F.leaky_relu(self.bn_b0(self.l_b0(z))),
                      (B, self.gf_dim, 4, 4))
        h = self.dc_b1(h)
        h = self.dc_b2(h)
        h = self.dc_b3(h)
        h_back = F.tanh(self.dc_b4(h))  ### (B, CH, Y, X)
        h_back = F.expand_dims(h_back, 2)
        h_back = F.tile(h_back, (1, 1, T, 1, 1))  ### tile to (B, CH, T, Y, X)

        ### fore img generation
        h_c = F.reshape(F.leaky_relu(self.bn_f0(self.l_f0(z))),
                        (B, self.gf_dim // 2, 2, 4, 4))
        h_c = self.dc_f1(h_c)
        h_c = self.dc_f2(h_c)

        ### colorize flow w U-net
        ## encode flow
        h = flow
        h_cm1 = self.c_m1(h)
        h_cm2 = self.c_m2(h_cm1)

        h = F.concat((h_cm2, h_c))

        h_cm3 = self.c_m3(h)

        h_cm4 = self.c_m4(h_cm3)
        h_cm5 = self.c_m5(h_cm4)

        ## decode
        h = self.dc_m1(h_cm5)
        h = self.dc_m2(F.concat((h, h_cm4)))
        h_dc3 = self.dc_m3(F.concat((h, h_cm3)))
        h = self.dc_m4(F.concat((h_dc3, h_cm1)))

        h_fore = F.tanh(self.dc_m5(h))

        ### make mask
        h_mask = F.sigmoid(self.dc_mask(h_dc3))
        h_mask = l1_penalty(h_mask, self.lamda)
        h_mask = F.tile(h_mask, (1, 3, 1, 1, 1))

        ### calc video
        x = h_mask * h_fore + (1 - h_mask) * h_back

        if chainer.config.train:
            return x
        else:
            return x, h_fore, h_back, h_mask

class Discriminator(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.Normal(0.02)
        super(Discriminator, self).__init__(
            c0_img=L.ConvolutionND(3, 3, 32, 4, 2, 1, initialW=w),
            c0_flow=L.ConvolutionND(3, 2, 32, 4, 2, 1, initialW=w),
            c1=L.ConvolutionND(3, 64, 128, 4, 2, 1, initialW=w),
            c2=L.ConvolutionND(3, 128, 256, 4, 2, 1, initialW=w),
            c3=L.ConvolutionND(3, 256, 512, 4, 2, 1, initialW=w),
            l4=L.Linear(None, 1, initialW=w),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x, flow):
        h_img = F.leaky_relu(self.c0_img(x))
        h_flow = F.leaky_relu(self.c0_flow(flow))
        h = F.concat((h_img, h_flow))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        return self.l4(h)

class GAN_Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen_flow, self.dis_flow, self.gen_tex, self.dis_tex = kwargs.pop('models')
        self.C = kwargs.pop('C')
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

    def loss_gen_multi(self, gen, y_flowgan_fake, y_texgan_fake, C=0.1):
        batchsize = y_flowgan_fake.data.shape[0]
        loss_flow = F.sum(F.softplus(-y_flowgan_fake)) / batchsize
        loss_tex = F.sum(F.softplus(-y_texgan_fake)) / batchsize
        loss = loss_flow + C * loss_tex
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_flowgan_optimizer = self.get_optimizer('gen_flow')
        dis_flowgan_optimizer = self.get_optimizer('dis_flow')
        gen_texgan_optimizer = self.get_optimizer('gen_tex')
        dis_texgan_optimizer = self.get_optimizer('dis_tex')

        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        video_real, flow_real = tuple(Variable(x) for x in in_arrays)
        gen_flow, dis_flow, gen_tex, dis_tex = self.gen_flow, self.dis_flow, self.gen_tex, self.dis_tex
        xp = chainer.cuda.get_array_module(video_real.data)
        batchsize = video_real.data.shape[0]

        ### flow GAN
        ## discriminate real flow
        y_flowgan_real = dis_flow(flow_real)
        ## generate fake flow
        z_flow = Variable(xp.asarray(gen_flow.make_hidden(batchsize)))
        flow_fake = gen_flow(z_flow)
        y_flowgan_fake = dis_flow(flow_fake)

        ### Texture GAN
        ## discriminate real video
        y_texgan_real = dis_tex(video_real, flow_real)

        ## generate fake video
        z_tex = Variable(xp.asarray(gen_tex.make_hidden(batchsize)))
        video_fake = gen_tex(z_tex, flow_fake)
        y_texgan_fake = dis_tex(video_fake, flow_fake)


        ### update
        ## video discriminator
        dis_texgan_optimizer.update(self.loss_dis, dis_tex, y_texgan_fake, y_texgan_real)
        ## video generator
        gen_texgan_optimizer.update(self.loss_gen, gen_tex, y_texgan_fake)
        ## flow discriminator
        dis_flowgan_optimizer.update(self.loss_dis, dis_flow, y_flowgan_fake, y_flowgan_real)
        ## flow generator
        gen_flowgan_optimizer.update(self.loss_gen_multi, gen_flow, y_flowgan_fake, y_texgan_fake, C=self.C)
