import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from dataset import path_config
from d2l import torch as d2l


nn_Module = nn.Module



class DatasetMulti(torch.utils.data.Dataset):
    def __init__(self, paths, *args, **kwargs):


        self.base_path = None

        self.datasets = []


        for cur_base_path in paths:
            if os.path.isdir(cur_base_path):
                for root, dirs, files in os.walk(cur_base_path):
                    for file in files:
                        if file.endswith(".pth"):
                            cur_path = os.path.join(cur_base_path, file)
                            dataset = Dataset(cur_path,*args, **kwargs)
                            if dataset.is_valid():
                                self.datasets.append(dataset)

            else:

                self.datasets.append(Dataset(cur_base_path,*args, **kwargs))


        dataset_sizes = [len(dataset) for dataset in self.datasets]
        self.num_elements = sum(dataset_sizes)
        self.cum_size = np.cumsum(np.array(dataset_sizes))

    def __getitem__(self, item_id):
        dataset_id = np.searchsorted(self.cum_size, item_id, side='right')

        if dataset_id > 0:
            item_id = item_id - self.cum_size[dataset_id-1]

        return self.datasets[dataset_id][item_id]

    def __len__(self):
        return self.num_elements

    def resolution(self):
        return self.datasets[0].resolution()



class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super(Dataset, self).__init__()

        if path_config.copy_dataset:
            import tempfile
            import shutil
            tfile_name =os.path.join(tempfile.gettempdir(), os.path.basename(path))
            shutil.copy2(path, tfile_name)
            print("Copying to ", tfile_name)
            path = tfile_name
        self.path = path
        dataset = self.open_dataset()
        self.num_elements = dataset["left_color"].shape[0]

    def open_dataset(self):
        return h5py.File(self.path, 'r')

    def __len__(self):
        return self.num_elements

    def resolution(self):
        dataset = self.open_dataset()
        return tuple(dataset["left_color"].shape[1:3])

    def get_height_width(self):

        dataset = self.open_dataset()

        height, width = dataset["left_color"].shape[1:3]

        return height, width

    def __getitem__(self, item_id):
        dataset = self.open_dataset()

        ground_color = dataset["left_color"][item_id, ...]
        ground_light = dataset["right_color"][item_id, ...]

        def convert_buffer(x):
            return torch.Tensor(x).float()#.permute([2,0,1])

        ground_colorgray=0.2126 * ground_color[:, :, 0] + 0.7152 * ground_color[:, :, 1] + 0.0722 * ground_color[:, :, 2]
        #ground_colorgray=ground_colorgray.reshape(512,512,1)
        #print(ground_color)
        #print(ground_color.shape,ground_colorgray.shape)
        ground_colorfft=np.fft.fft2(ground_colorgray)
        #print(ground_colorfft.shape)
        #ground_colorfft=np.log(ground_colorfft)
        #ground_colorfft=ground_colorfft.reshape(512,512,1)
        ground_colorfft1=np.real(ground_colorfft)
        ground_colorfft1=ground_colorfft1/ground_colorfft1.max()
        #print("dsa",ground_colorfft1.max())
        #sss
        #ground_colorfft1=ground_colorfft1/ground_colorfft1.
        ground_colorfft2=np.real(ground_colorfft)
        ground_colorfft2=ground_colorfft2/ground_colorfft2.max()
        #print(ground_colorfft2)
        #asdsadas
        ground_colorfft1 = ground_colorfft1.reshape(512, 512, 1)
        ground_colorfft2 = ground_colorfft2.reshape(512, 512, 1)
        #print(ground_color.shape,ground_colorfft1.shape,ground_colorfft2.shape)

        ground_color=np.concatenate([ground_color,ground_colorfft1,ground_colorfft2],axis=-1)
        #print("ground_color",ground_color.shape)
        #print(ground_color.shape, ground_colorgray.shape)
        #print("1",ground_color)


        ground_color = convert_buffer(ground_color)
        #print("2",ground_color.shape)

        ground_color = pepepe(ground_color)
        #print("2", ground_color.shape)
        ground_color=ground_color.permute(2,0,1)
        #print("2", ground_color.shape)
        ground_light = convert_buffer(ground_light)
        #print("ground_camera_dir.shape", ground_camera_dir.shape)
        ground_light = ground_light.permute(2, 0, 1)



        input = ground_color
        target=ground_light
        return input, target

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        # print((len(self.funcs) * N_freqs + 1))
        # print((len(self.funcs)),N_freqs)
        #
        # print(self.in_channels)
        # print(self.out_channels)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)
dirEb = Embedding(4, 20)
def pepepe(X):
    X1 = dirEb(X)
    return(X1)
from positionembeding import Embedding




class Residual(nn.Module):  #@save
    def __init__(self, input_channels=256, num_channels=256,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = Inception(input_channels, 64, (96, 128), (16, 32), 32)
        self.conv2 = Inception(256, 64, (96, 128), (16, 32), 32)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(256, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        #print("reX",X.shape,X)
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def loaddataset(args):
    dataset_paths = [path_config.get_value("path_dataset", args.dataset)]
    print(dataset_paths)
    #dataset = DatasetMulti(dataset_paths)
    dataset=DatasetMulti(['/mnt/d/doubledataset/Holopix50k2.hdf5'])
    train_size = int(len(dataset) * 0.9)
    test_size = int(len(dataset) -train_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size],generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=8)
    return train_loader,test_loader

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

class savednetwork():
    def __init__(self, args):
        self.args=args
        self.input_channels = 205
        self.num_epochs = self.args.epoch
        self.epoch = 0
        self.save_every = 1
        self.devices = [d2l.try_gpu(i) for i in range(self.args.num_gpus)]
        self.batch_size = self.args.batch
        self.num_gpus = self.args.num_gpus
        self.batch = self.args.batch
        self.lr = self.args.lr
        self.outm = self.args.outm

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.normal_(m.weight, std=0.01)

        self.net = nn.Sequential(
            Residual(self.input_channels),
            Residual(256,256),
            Residual(256,256),
            Residual(256,256),
            Residual(256,256),
            torch.nn.Conv2d(256, 3, 1)
        )
        self.net.apply(init_weights)
        # 在多个GPU上设置模型
        self.net =self.net.to(self.devices[0])
        self.net = nn.DataParallel(self.net, device_ids=self.devices)

        self.trainer = torch.optim.Adam(self.net.parameters(), self.args.lr)
        self.loss = nn.MSELoss()

        self.path_save = os.path.join(path_config.path_models, self.args.outm)


class bbbbxxxx():
    def __init__(self, args):
        self.args = args
        self.batch_size = self.args.batch
        self.num_epochs = self.args.epoch
        self.num_gpus = self.args.num_gpus
        self.batch = self.args.batch
        self.lr = self.args.lr
        self.outm = self.args.outm
        self.devices = [d2l.try_gpu(i) for i in range(self.args.num_gpus)]
        print(self.devices)

        ##################
        self.vars = savednetwork(args)
        if self.args.inm:
            path_load = path_config.get_value("path_models", self.args.inm)

            self.vars.load(path_load)

    def train(self):
        train_iter, test_iter = loaddataset(self.vars.args)
        print("load data")
        self.timer, self.num_epochs = d2l.Timer(), self.num_epochs

        self.animator = d2l.Animator('epoch', 'test acc', xlim=[1, self.num_epochs])
        print("prepare to start raining")

        for self.vars.epoch in range(self.num_epochs):

            self.vars.net.train()
            self.timer.start()
            print("start",self.vars.epoch,"epoch")
            for X, y in train_iter:
                self.vars.trainer.zero_grad()
                X, y = X.to(self.devices[0]), y.to(self.devices[0])
                print(X,X.shape)
                l = self.vars.loss(self.vars.net(X), y)
                l.backward()
                self.vars.trainer.step()
                print("finish one time train at ", self.vars.epoch)
            self.timer.stop()
            self.vars.animator.add(self.vars.epoch + 1, (d2l.evaluate_accuracy_gpu(self.vars.net, test_iter),))
            if self.vars.epoch % self.save_every == 0:
                self.vars.save(self.path_save)
            print("finish train at ",self.vars.epoch)
        print(f'测试精度：{self.vars.animator.Y[0][-1]:.2f}，{self.vars.timer.avg():.1f}秒/轮，'
              f'在{str(self.vars.devices)}')


        self.vars.save(self.path_save)

    def save(self, path):
        data = {}

        for name in vars(self):
            if name not in {"live"}:
                data[name] = getattr(self, name)
        #print(data, dir(data))
        print(data.items())

        torch.save(data, path + "~")
        os.rename(path + "~", path)

    def load(self, path):
        data = torch.load(path)
        # print(dir(self))
        for key, value in data.items():
            setattr(self, key, value)
            # print(dir(self))
            # if hasattr(self, key):
            #     setattr(self, key, value)
            # else:
            #     print("No '{}' in the class".format(key))
        # print(dir(self))




def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inm', help='input model path', type=str, default=None)
    parser.add_argument('--outm', help='output model path', type=str, default="default.pth")


    parser.add_argument('--export', help='output model path', type=str, default=None)


    parser.add_argument('--batch', default=1, type=int)


    parser.add_argument('--dataset', help='dataset path', type=str, default="holopix50k2.hdf5")

    parser.add_argument('--loss', default='mse', type=str)

    parser.add_argument('--epoch', help='out name', type=int, default=100)
    parser.add_argument('--lr', help='Color boost', type=float, default=5e-4)
    parser.add_argument('--num_gpus', help='Color boost', type=int, default=1)

    parser.add_argument('--workers', help='out name', type=int, default=0)



    args = parser.parse_args()

    neralnet = bbbbxxxx(args)
    print("call train function")

    neralnet.train()







if __name__ == "__main__":
    main()
