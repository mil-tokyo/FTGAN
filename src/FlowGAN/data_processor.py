
from __future__ import print_function

import chainer
import random
import numpy as np
import scipy.misc
import os

class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, root_dir, crop_size=(64, 64), video_len=32):
        self.dataset = dataset
        self.root_dir = root_dir
        self.crop_x = crop_size[0]
        self.crop_y = crop_size[1]
        self.video_len = video_len
        self.stride = self.video_len//2

    def __len__(self):
        return len(self.dataset)

    def preprocess_img(self, flow, flipcrop=True):
        crop_x = self.crop_x
        crop_y = self.crop_y

        _, _, h, w = flow.shape

        if flipcrop:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_y - 1)
            left = random.randint(0, w - crop_x - 1)
            if random.randint(0, 1):
                flow = flow[:, :, :, ::-1]
                flow[0,:,:,:] = 255 - flow[0,:,:,:]
            bottom = top + crop_y
            right = left + crop_x
            flow = flow[:,:, top:bottom, left:right]
        else:
            dst_flow = np.zeros((2,flow.shape[1],self.crop_y, self.crop_x), np.uint8)
            for i in range(flow.shape[1]):
                dst_flow[0, i] = scipy.misc.imresize(flow[0, i, :, :], [self.crop_y, self.crop_x],'bicubic')
                dst_flow[1, i] = scipy.misc.imresize(flow[1, i, :, :], [self.crop_y, self.crop_x],'bicubic')
            flow = dst_flow
        flow = flow.astype(np.float32) * (2 / 255.) - 1.
        return flow

    def get_example(self, i, train=True):
        while 1:
            if os.path.exists(self.root_dir + self.dataset[i] + '.npy'):
                video_flow = np.load(self.root_dir + self.dataset[i] + '.npy')
                N = video_flow.shape[0]
                try:
                    if N == 32:
                        j = 0
                    else:
                        j = np.random.randint(0, N - self.video_len)
                except:
                    i = np.random.randint(len(self.dataset))
                    continue
                break
            else:
                print('no data',self.dataset[i])
                i = np.random.randint(len(self.dataset))

        Flows = video_flow[j:j+self.video_len].transpose(3, 0, 1, 2)  ## (T,Y,X,ch) -> (ch,T,Y,X)
        flow = self.preprocess_img(Flows, flipcrop=True)

        return flow
