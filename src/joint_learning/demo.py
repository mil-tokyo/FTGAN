#!/usr/bin/env python
"""
Code for reproducing main results in the paper
Hierarchical Video Generation from Orthogonal Information: Optical Flow and Texture (AAAI-18)
https://arxiv.org/abs/1711.09618
Katsunori Ohnishi*, Shohei Yamamoto*, Yoshitaka Ushiku, Tatsuya Harada.
Note that * indicates equal contribution.
"""
from __future__ import print_function
import argparse

import matplotlib
from chainer import Variable

# Disable interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import chainer
from chainer import cuda
from chainer import serializers
import net

def gan_test(args, model_path):
    # Prepare Flow and Texture GAN model, defined in net.py

    gen_flow = net.FlowGenerator()
    serializers.load_npz(model_path["gen_flow"], gen_flow)
    gen_tex = net.Generator(dimz=100)
    serializers.load_npz(model_path["gen_tex"], gen_tex)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen_flow.to_gpu()
        gen_tex.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    rows = 5
    cols = 5

    ### generate videos from Z
    np.random.seed(0)
    for i in range(10):
        print(i)
        z_flow = Variable(xp.asarray(gen_flow.make_hidden(rows * cols)))
        z_tex = Variable(xp.asarray(gen_tex.make_hidden(rows * cols)))

        ### generate flow
        with chainer.using_config('train', False):
            flow_fake, _, _ = gen_flow(z_flow)
        flow_fake_tmp = chainer.cuda.to_cpu(flow_fake.data)

        ### generate video
        with chainer.using_config('train', False):
            y, fore_vid, back_img, h_mask = gen_tex(z_tex, flow_fake)
        y = chainer.cuda.to_cpu(y.data)
        fore_vid = chainer.cuda.to_cpu(fore_vid.data)
        back_img = chainer.cuda.to_cpu(back_img.data)
        y_mask = chainer.cuda.to_cpu(h_mask.data)
        flow = flow_fake_tmp

        preview_dir = '{}/{:03}/'.format(args.out, i)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        ## save video
        y = np.asarray(np.clip((y + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape
        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch
        for j in range(0, T):
            preview_path = preview_dir + 'img_{:03}.jpg'.format(j + 1)
            Image.fromarray(Y[j]).save(preview_path)

        ### save fore video
        y = np.asarray(np.clip((fore_vid + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape
        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch
        for j in range(0, T):
            preview_path = preview_dir + 'fore_{:03}.jpg'.format(j + 1)
            Image.fromarray(Y[j]).save(preview_path)

        ### save mask video
        y = np.asarray(np.clip(y_mask*255., 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape
        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch
        for j in range(0, T):
            preview_path = preview_dir + 'mask_{:03}.jpg'.format(j + 1)
            Image.fromarray(Y[j]).save(preview_path)

        ### save back img
        y = np.asarray(np.clip((back_img + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape
        y = y[:, :, 0]
        Y = y.reshape((rows, cols, CH, H, W))
        Y = Y.transpose(0, 3, 1, 4, 2)  ### rows, H, cols, W, ch
        Y = Y.reshape((rows * H, cols * W, CH))  # T, H, W, ch
        preview_path = preview_dir + 'back.jpg'
        Image.fromarray(Y).save(preview_path)

        ### save flow
        y = np.asarray(np.clip((flow + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape
        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch

        for j in range(0, T):
            preview_path = preview_dir + 'flow_{:03}.jpg'.format(j + 1)
            flow_img = np.hstack((Y[j, :, :, 0], Y[j, :, :, 1]))
            Image.fromarray(flow_img).save(preview_path)

def main():
    parser = argparse.ArgumentParser(description='Hierarchical Video Generation from Orthogonal Information: Optical Flow and Texture (AAAI-18)')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='../result_demo/',
                        help='Output directory')
    parser.add_argument('--loaderjob', '-j', type=int, default=8,
                        help='Number of parallel data loading processes')
    parser.add_argument('--video_len', '-vl', type=int, default=32,
                        help='video length (default: 32)')
    parser.add_argument('--flow_random', '-fr', type=int, default=1,
                        help='')
    parser.add_argument('--modeldir', '-md', type=str,default='../../models/',
                        help='')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('FlowGAN model : {}'.format(args.modeldir + 'gen_flow_iteration_10000.npz'))
    print('TextureGAN model : {}'.format(args.modeldir + 'gen_tex_iteration_10000.npz'))

    model_path = {
        "gen_flow": args.modeldir + 'gen_flow_iteration_10000.npz',
        "gen_tex": args.modeldir + 'gen_tex_iteration_10000.npz',
        }

    ## main test
    gan_test(args, model_path)

if __name__ == '__main__':
    main()
