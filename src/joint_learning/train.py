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

import visualizer
from data_processor import PreprocessedDataset
from net import GAN_Updater
import net


# Setup optimizer
def make_optimizer(model, args, alpha=1e-6, beta1=0.5, beta2=0.999, epsilon=1e-8):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2, eps=epsilon)
    optimizer.setup(model)
    if args.weight_decay:
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
    return optimizer


def gan_training(args, train, test, model_path):
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    if args.loaderjob:
        train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize, n_processes=args.loaderjob)
    else:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Prepare Flow and Texture GAN model, defined in net.py
    gen_flow = net.FlowGenerator()
    dis_flow = net.FlowDiscriminator()
    gen_tex = net.Generator(args.dimz)
    dis_tex = net.Discriminator()

    serializers.load_npz(model_path["gen_flow"], gen_flow)
    serializers.load_npz(model_path["dis_flow"], dis_flow)
    serializers.load_npz(model_path["gen_tex"], gen_tex)
    serializers.load_npz(model_path["dis_tex"], dis_tex)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen_flow.to_gpu()
        dis_flow.to_gpu()
        gen_tex.to_gpu()
        dis_tex.to_gpu()

    xp = np if args.gpu < 0 else cuda.cupy

    opt_flow_gen = make_optimizer(gen_flow, args, alpha=1e-7)
    opt_flow_dis = make_optimizer(dis_flow, args, alpha=1e-7)
    opt_tex_gen = make_optimizer(gen_tex, args, alpha=1e-6)
    opt_tex_dis = make_optimizer(dis_tex, args, alpha=1e-6)

    # Updater
    updater = GAN_Updater(
        models=(gen_flow, dis_flow, gen_tex, dis_tex),
        iterator=train_iter,
        optimizer={'gen_flow': opt_flow_gen, 'dis_flow': opt_flow_dis, 'gen_tex': opt_tex_gen, 'dis_tex': opt_tex_dis},
        device=args.gpu, C=args.C)

    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    snapshot_interval = (args.snapshot_interval), 'iteration'
    visualize_interval = (args.visualize_interval), 'iteration'
    log_interval = (args.log_interval), 'iteration'

    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)

    trainer.extend(extensions.LogReport(trigger=log_interval))

    trainer.extend(extensions.PlotReport(['gen_flow/loss', 'dis_flow/loss', 'gen_tex/loss', 'dis_tex/loss'], trigger=log_interval, file_name='plot.png'))

    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen_flow/loss', 'dis_flow/loss', 'gen_tex/loss', 'dis_tex/loss'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen_flow, 'gen_flow_iteration_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis_flow, 'dis_flow_iteration_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen_tex, 'gen_tex_iteration_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis_tex, 'dis_tex_iteration_{.updater.iteration}.npz'), trigger=snapshot_interval)


    trainer.extend(visualizer.extension(test, (gen_flow, gen_tex),  args, rows=args.rows, cols=args.cols),
                   trigger=visualize_interval)

    if args.adam_decay_iteration:
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt_flow_gen),
                       trigger=(args.adam_decay_iteration, 'iteration'))
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt_flow_dis),
                       trigger=(args.adam_decay_iteration, 'iteration'))
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt_tex_gen),
                       trigger=(args.adam_decay_iteration, 'iteration'))
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt_tex_dis),
                       trigger=(args.adam_decay_iteration, 'iteration'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)


    # Run the training
    trainer.run()


def main():
    parser = argparse.ArgumentParser(description='Hierarchical Video Generation from Orthogonal Information: Optical Flow and Texture (AAAI-18)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--iteration', '-e', default=12000, type=int,
                        help='number of iterations (default: 12000)')
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
    parser.add_argument('--out', '-o', default='../result_joint/',
                        help='Output directory')
    parser.add_argument('--loaderjob', '-j', type=int, default=8,
                        help='Number of parallel data loading processes')
    parser.add_argument('--adam_decay_iteration', '-ade', type=int, default=2000,
                        help='adam decay iteration (default: 2000)')
    parser.add_argument('--video_len', '-vl', type=int, default=32,
                        help='video length (default: 32)')
    parser.add_argument('--rows', type=int, default=5,
                        help='rows')
    parser.add_argument('--cols', type=int, default=5,
                        help='cols')
    parser.add_argument('--weight_decay', '-wd', type=int, default=0,
                        help='')
    parser.add_argument('--C', '-c', type=float, default=0.1,
                        help='')

    parser.add_argument('--flowgan', type=str,default='../result_flowgan/',
                        help='path to FlowGAN model')
    parser.add_argument('--texgan', type=str,default='../result_texgan/',
                        help='path to TextureGAN model')

    parser.add_argument('--root', type=str, default='',
                        help='dataset directory')
    parser.add_argument('--dataset', '-d', type=str,default='penn_action',
                        help='')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# iteration: {}'.format(args.iteration))
    print('# dataset: {}'.format(args.dataset))

    npy_root  = args.root + '/npy_76/'
    flow_root = args.root + '/npy_flow_76/'

    Train = []
    f = open('../../data/penn_action/train.txt')
    for line in f.readlines():
        Train.append(line.split()[0])
    f.close()

    Test = []
    f = open('../../data/penn_action/test.txt')
    for line in f.readlines():
        Test.append(line.split()[0])
    f.close()

    train = PreprocessedDataset(Train, npy_root, flow_root, video_len=args.video_len)
    test  = PreprocessedDataset(Test,  npy_root, flow_root, video_len=args.video_len)

    model_path = {
        "gen_flow": args.flowgan + 'gen_iteration_60000.npz',
        "dis_flow": args.flowgan + 'dis_iteration_60000.npz',
        "gen_tex":  args.texgan  + 'gen_iteration_60000.npz',
        "dis_tex":  args.texgan  + 'dis_iteration_60000.npz'}

    ## main training
    gan_training(args, train, test, model_path)

if __name__ == '__main__':
    main()
