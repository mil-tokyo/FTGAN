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
import six
import h5py
import os
from PIL import Image
import pickle
import random
import sys

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer.training import extensions
from chainer import training

from visualizer import extension
from data_processor import PreprocessedDataset
from net import GAN_Updater
import net

def gan_training(args, test, model_path):
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    # Prepare GANGAN model, defined in net.py

    gen_flow = net.FlowGenerator()
    serializers.load_npz(model_path["gen_flow"], gen_flow)
    gen_tex = net.Generator(args.dimz)
    serializers.load_npz(model_path["gen_tex"], gen_tex)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen_flow.to_gpu()
        gen_tex.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    rows = 1
    cols = 1

    ### generate videos from Z
    np.random.seed(0)
    if args.flow_random == False:
        indices = np.random.randint(0, len(test), rows * cols).tolist()

    for i in range(50):
        print(i)
        if args.flow_random:
            indices = np.random.randint(0, len(test), rows * cols).tolist()

        #RawVideos, RawFlows = test.get_example4test(indices, True)
        #flow_real = Variable(xp.asarray(RawFlows), volatile=True)
        z_flow = Variable(xp.asarray(gen_flow.make_hidden(rows * cols)), volatile=True)
        flow_fake,_,_ = gen_flow(z_flow, test=True)
        z = Variable(xp.asarray(gen_tex.make_hidden(rows * cols)), volatile=True)
        y, h_fore, h_back, h_mask = gen_tex(z, flow_fake, test=True)
        y = chainer.cuda.to_cpu(y.data)
        y_fore = chainer.cuda.to_cpu(h_fore.data)
        y_back = chainer.cuda.to_cpu(h_back.data)
        y_mask = chainer.cuda.to_cpu(h_mask.data)

        preview_dir = '{}/preview/{:03}/'.format(args.out, i)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        ## save whole images
        y = np.asarray(np.clip((y + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape

        Y = y.reshape((rows, cols, 3, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch
        for i in range(0, T):
            preview_path = preview_dir + 'img_{:03}.jpg'.format(i + 1)
            Image.fromarray(Y[i]).save(preview_path)

        ### save fore videos
        y = np.asarray(np.clip((y_fore + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape

        Y = y.reshape((rows, cols, 3, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch
        for i in range(0, T):
            preview_path = preview_dir + 'fore_{:03}.jpg'.format(i + 1)
            Image.fromarray(Y[i]).save(preview_path)

        ### save mask videos
        y = np.asarray(np.clip(y_mask * 255., 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape

        Y = y.reshape((rows, cols, CH, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch
        Y = Y[:, :, :, 0]
        for i in range(0, T):
            preview_path = preview_dir + 'mask_{:03}.jpg'.format(i + 1)
            Image.fromarray(Y[i]).save(preview_path)

        ### save back
        y = np.asarray(np.clip((y_back + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        B, CH, T, H, W = y.shape

        Y = y.reshape((rows, cols, 3, T, H, W))
        Y = Y.transpose(3, 0, 4, 1, 5, 2)  ### T, rows, H, cols, W, ch
        Y = Y.reshape((T, rows * H, cols * W, CH))  # T, H, W, ch
        preview_path = preview_dir + 'back.jpg'
        Image.fromarray(Y[0]).save(preview_path)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: GAN on miku')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--iteration', '-e', default=60000, type=int,
                        help='number of iterations (default: 100000)')
    parser.add_argument('--dimz', '-z', default=100, type=int,
                        help='dimention of encoded vector (default: 100)')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='learning minibatch size (default: 32)')
    parser.add_argument('--log_interval', '-li', type=int, default=100,
                        help='log interval iter (default: 100)')
    parser.add_argument('--snapshot_interval', '-si', type=int, default=10000,
                        help='snapshot interval iter (default: 10000)')
    parser.add_argument('--visualize_interval', '-vi', type=int, default=1000,
                        help='visualize interval iter (default: 1000)')
    parser.add_argument('--out', '-o', default='../result_joint_test/',
                        help='Output directory')
    parser.add_argument('--loaderjob', '-j', type=int, default=8,
                        help='Number of parallel data loading processes')
    parser.add_argument('--adam_decay_iteration', '-ade', type=int, default=10000,
                        help='adam decay iteration (default: 0)')
    parser.add_argument('--video_len', '-vl', type=int, default=32,
                        help='video length (default: 32)')
    parser.add_argument('--weight_decay', '-wd', type=int, default=0,
                        help='')
    parser.add_argument('--flow_random', '-fr', type=int, default=1,
                        help='')

    parser.add_argument('--root', type=str, default='',
                        help='dataset directory')
    parser.add_argument('--dataset', '-d', type=str,
                        help='')

    parser.add_argument('--modeldir', '-md', type=str,default='../result_joint/'
                        help='')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# iteration: {}'.format(args.iteration))
    print('# dataset'.format(args.dataset))

    npy_root  = args.root + '/npy_76/'
    flow_root = args.root + '/npy_flow_76/'

    Test = []
    f = open('../../data/penn_action/test.txt')
    for line in f.readlines():
        Test.append(line.split()[0])
    f.close()

    test = PreprocessedDataset(Test, npy_root, flow_root, video_len=args.video_len)

    model_path = {
        "gen_flow": args.modeldir + '/gen_flow_iteration_10000.npz',
        "dis_flow": args.modeldir + '/dis_flow_iteration_10000.npz',
        "gen_tex": args.modeldir + '/gen_tex_iteration_10000.npz',
        "dis_tex": args.modeldir + '/dis_tex_iteration_10000.npz',
        }
        
    ## main test
    gan_training(args, test, model_path)

if __name__ == '__main__':
    main()
