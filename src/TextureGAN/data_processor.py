import chainer
import numpy as np
import random, os
import scipy.misc

class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, video_path, flow_path, video_len=32, crop_size=(64,64)):
        self.dataset = dataset
        self.flow_path = flow_path
        self.video_path = video_path
        self.video_len = video_len
        self.crop_y = crop_size[0]
        self.crop_x = crop_size[1]
        self.stride = self.video_len//2

    def __len__(self):
        return len(self.dataset)

    def preprocess_img(self, image, flow, flipcrop=True):
        crop_x = self.crop_x
        crop_y = self.crop_y
        _, _, h, w = image.shape

        if flipcrop:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_y - 1)
            left = random.randint(0, w - crop_x - 1)
            if random.randint(0, 1):
                image = image[:, :, :, ::-1]
                flow = flow[:, :, :, ::-1]
                flow[0,:,:,:] = 255 - flow[0,:,:,:]

            bottom = top + crop_y
            right = left + crop_x
            image = image[:, :, top:bottom, left:right]
            flow = flow[:, :, top:bottom, left:right]
        else:
            dst_flow = np.zeros((2, flow.shape[1], self.crop_y, self.crop_x), np.uint8)
            for i in range(flow.shape[1]):
                dst_flow[0, i] = scipy.misc.imresize(flow[0, i, :, :], [self.crop_y, self.crop_x], 'bicubic')
                dst_flow[1, i] = scipy.misc.imresize(flow[1, i, :, :], [self.crop_y, self.crop_x], 'bicubic')
            flow = dst_flow
            dst_image = np.zeros((3, image.shape[1], self.crop_y, self.crop_x), np.uint8)
            for i in range(image.shape[1]):
                dst_image[:,i] = scipy.misc.imresize(image[:, i, :, :].transpose(1,2,0), [self.crop_y, self.crop_x], 'bicubic').transpose(2,0,1)
            image = dst_image

        image = image.astype(np.float32) * (2 / 255.) - 1.
        flow = flow.astype(np.float32)  * (2 / 255.) - 1.
        return image, flow


    def convertflow(self, flow):
        CH, _, Y, X = flow.shape
        T = self.video_len
        dst_flow = np.zeros((CH, self.flow_len, T, Y, X), np.float32)
        for i in range(T):
            dst_flow[ :, :, i, :, :] = flow[ :, i:i + self.flow_len, :, :]
        return dst_flow.reshape((-1, T, Y, X))

    def get_example(self, i, train=True):
        while 1:
            if os.path.exists(self.video_path + self.dataset[i] + '.npy'):
                video_rgb = np.load(self.video_path + self.dataset[i] + '.npy')
                video_flow = np.load(self.flow_path + self.dataset[i] + '.npy')
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

        Imgs = video_rgb[j:j+self.video_len].transpose(3, 0, 1, 2)  ## (T,Y,X,ch) -> (ch,T,Y,X)
        Flows = video_flow[j:j+self.video_len].transpose(3, 0, 1, 2)  ## (T,Y,X,ch) -> (ch,T,Y,X)
        img, flows = self.preprocess_img(Imgs, Flows, True)

        return img, flows

    def get_example4test(self, test_indices, flag=False):
        if not hasattr(self, 'Videos') or flag:
            self.test_indices = test_indices
            video_sample, flow_sample = self.get_example(0)
            Videos = np.zeros((len(self.test_indices),) + video_sample.shape, np.float32)
            Flows = np.zeros((len(self.test_indices),) + flow_sample.shape, np.float32)
            for (i, id) in enumerate(self.test_indices):
                Videos[i], Flows[i] = self.get_example(id)

            self.Videos = Videos
            self.Flows = Flows
        return self.Videos, self.Flows
