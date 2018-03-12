# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image

import chainer
from chainer import cuda
from chainer import Variable

def extension(data_processor, model, indices, args, data_type, rows=5, cols=5, seed=0):
    @chainer.training.make_extension()
    def make_video(trainer):
        ### prepare

        xp = cuda.cupy
        RawVideos, RawFlows = data_processor.get_example4test(indices)
        flow_real = Variable(xp.asarray(RawFlows))

        np.random.seed(seed)
        z_app = Variable(xp.asarray(model.make_hidden(rows * cols)))
        np.random.seed()

        ### generate video
        with chainer.using_config('train', False):
            y, fore_vid, back_img, h_mask = model(z_app, flow_real)
        y = chainer.cuda.to_cpu(y.data)
        fore_vid = chainer.cuda.to_cpu(fore_vid.data)
        back_img = chainer.cuda.to_cpu(back_img.data)
        y_mask = chainer.cuda.to_cpu(h_mask.data)


        preview_dir = '{}/preview/'.format(args.out) + str(trainer.updater.iteration) + '/{}/'.format(data_type)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        ## save video
        y = np.asarray(np.clip((y + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape
        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch
        for i in range(0, T):
            preview_path = preview_dir + 'img_{:03}.jpg'.format(i + 1)
            Image.fromarray(Y[i]).save(preview_path)

        ### save fore video
        y = np.asarray(np.clip((fore_vid + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape
        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch
        for i in range(0, T):
            preview_path = preview_dir + 'fore_{:03}.jpg'.format(i + 1)
            Image.fromarray(Y[i]).save(preview_path)

        ### save mask video
        y = np.asarray(np.clip(y_mask*255., 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape
        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch
        for i in range(0, T):
            preview_path = preview_dir + 'mask_{:03}.jpg'.format(i + 1)
            Image.fromarray(Y[i]).save(preview_path)

        ### save back img
        y = np.asarray(np.clip((back_img + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape
        y = y[:, :, 0]
        Y = y.reshape((rows, cols, CH, H, W))
        Y = Y.transpose(0, 3, 1, 4, 2)  ### rows, H, cols, W, ch
        Y = Y.reshape((rows * H, cols * W, CH))  # T, H, W, ch
        preview_path = preview_dir + 'back.jpg'
        Image.fromarray(Y).save(preview_path)

        ### save raw video
        raw_dir = '{}/preview/raw/{}/'.format(args.out, data_type)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
            y = np.asarray(np.clip((RawVideos + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
            B, Ch, T, H, W = y.shape

            Y = y.reshape((rows, cols, CH, T, H, W))
            Y = Y.transpose(3, 0, 4, 1, 5, 2)
            Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch

            ## save whole images
            for i in range(T):
                preview_path = raw_dir + 'img_{:03}.jpg'.format(i + 1)
                Image.fromarray(Y[i]).save(preview_path)
            RawImg = Y[:]

            y = np.asarray(np.clip((RawFlows + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
            B, CH, T, H, W = y.shape

            Y = y.reshape((rows, cols, CH, T, H, W))  ###
            Y = Y.transpose(3, 0, 4, 1, 5, 2)  ## (rows, cols, T, ch, Y, X) -> (T, rows, Y, cols, X, ch)
            Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch

            ## save whole images
            for i in range(T):
                flow = Y[i]#.reshape((rows * H, cols * W, 2))
                flow_x = flow[:, :, 0]
                flow_y = flow[:, :, 1]

                preview_path = raw_dir + 'both_{:03}.jpg'.format(i + 1)
                flow_img = np.hstack((flow_x, flow_y))
                flow_img = np.tile(flow_img[:, :, np.newaxis], (1, 1, 3))
                Image.fromarray(np.hstack((RawImg[i], flow_img))).save(preview_path)

    return make_video
