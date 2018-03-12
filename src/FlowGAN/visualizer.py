# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image

import chainer
from chainer import cuda
from chainer import Variable


def extension(model, args, rows=5, cols=5, seed=0):
    @chainer.training.make_extension()
    def make_video(trainer):

        ### generate videos from Z
        xp = cuda.cupy
        np.random.seed(seed)
        z = Variable(xp.asarray(model.make_hidden(rows * cols)))

        with chainer.using_config('train', False):
            y, h_fore, h_mask = model(z)
        y = chainer.cuda.to_cpu(y.data)
        y_fore = chainer.cuda.to_cpu(h_fore.data)
        y_mask = chainer.cuda.to_cpu(h_mask.data)

        ## batch, ch, T, H, W

        ### save videos
        preview_dir = '{}/preview/'.format(args.out) + str(trainer.updater.iteration) + '/'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        ### save flow
        y = np.asarray(np.clip((y + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape
        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch

        for i in range(0, T):
            preview_path = preview_dir + 'flow_{:03}.jpg'.format(i + 1)
            flow_img = np.hstack((Y[i, :, :, 0], Y[i, :, :, 1]))
            Image.fromarray(flow_img).save(preview_path)

        ### save flow
        y = np.asarray(np.clip((y_fore + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y_fore.shape
        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch

        for i in range(0, T):
            preview_path = preview_dir + 'fore_{:03}.jpg'.format(i + 1)
            flow_img = np.hstack((Y[i, :, :, 0], Y[i, :, :, 1]))
            Image.fromarray(flow_img).save(preview_path)

        ### save mask
        y = np.asarray(np.clip(y_mask * 255., 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y_mask.shape
        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch

        for i in range(0, T):
            preview_path = preview_dir + 'mask_{:03}.jpg'.format(i + 1)
            flow_img = np.hstack((Y[i]))
            Image.fromarray(flow_img).save(preview_path)

    return make_video
