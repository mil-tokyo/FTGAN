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
import os

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer.training import extensions
from chainer import training

from visualizer import extension
from data_processor import PreprocessedDataset
from net import GAN_Updater
import net


# Setup optimizer
def make_optimizer(model, args, alpha=2e-4, beta1=0.5, beta2=0.999, epsilon=1e-8):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2, eps=epsilon)
    optimizer.setup(model)
    if args.weight_decay:
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
    return optimizer


def gan_training(args, train):
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    if args.loaderjob:
        train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize, n_processes=args.loaderjob)
    else:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Prepare Flow GAN model, defined in net.py
    gen = net.Generator(video_len=args.video_len)
    dis = net.Discriminator()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    opt_gen = make_optimizer(gen, args)
    opt_dis = make_optimizer(dis, args)

    # Updater
    updater = GAN_Updater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu)

    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    snapshot_interval = (args.snapshot_interval), 'iteration'
    visualize_interval = (args.visualize_interval), 'iteration'
    log_interval = (args.log_interval), 'iteration'

    trainer.extend(extensions.LogReport(trigger=log_interval))

    trainer.extend(extensions.PlotReport(['gen/loss', 'dis/loss'], trigger=log_interval, file_name='plot.png'))

    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss'
    ]), trigger=log_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)

    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iteration_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iteration_{.updater.iteration}.npz'), trigger=snapshot_interval)

    trainer.extend(extension(gen, args), trigger=visualize_interval)

    if args.adam_decay_iteration:
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt_gen),
                       trigger=(args.adam_decay_iteration, 'iteration'))
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt_dis),
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
    parser.add_argument('--iteration', '-i', default=60000, type=int,
                        help='number of iterations (default: 60000)')
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
    parser.add_argument('--out', '-o', default='../result_flowgan/',
                        help='Output directory')
    parser.add_argument('--loaderjob', '-j', type=int, default=8,
                        help='Number of parallel data loading processes')
    parser.add_argument('--adam_decay_iteration', '-adi', type=int, default=10000,
                        help='adam decay iteration (default: 10000)')
    parser.add_argument('--video_len', '-vl', type=int, default=32,
                        help='video length (default: 32)')
    parser.add_argument('--weight_decay', '-wd', type=int, default=0,
                        help='')

    parser.add_argument('--root', type=str, default='',
                        help='dataset directory')
    parser.add_argument('--dataset', '-d', type=str,default='penn_action',
                        help='')

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.system('mkdir -p ' + args.out)

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# iteration: {}'.format(args.iteration))
    print('# dataset: {}'.format(args.dataset))


    flow_root = args.root + '/npy_flow_76/'
    Train = []
    f = open('../../data/penn_action/train.txt')
    for line in f.readlines():
        Train.append(line.split()[0])
    f.close()
    train = PreprocessedDataset(Train, flow_root, video_len=args.video_len)

    ## main training
    gan_training(args, train)

if __name__ == '__main__':
    main()
