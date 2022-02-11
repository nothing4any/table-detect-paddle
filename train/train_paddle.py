#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2022/1/31 下午12:48
@file: table_ceil_paddle
@author: nothing4any
"""

import os
import sys
import time

import paddle

sys.path.append('.')
from table_line_paddle import Unet
from table_line_paddle import model
from glob import glob
from image_paddle import get_img_label
import subprocess
from sklearn.model_selection import train_test_split
from paddle.io import Dataset
import numpy as np

import paddle.fluid.dataloader.dataloader_iter

dir_path = os.path.dirname(os.path.abspath(__file__))
path = "/".join(dir_path.split("/")[:-1])
paddle.disable_static()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def cmd(command: str):
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    subp.wait(2)
    if subp.poll() == 0:
        return str(subp.communicate()[1])
    else:
        return None


def chick_gpu():
    command = "/opt/python-3.7/bin/pip list |grep paddlepaddle-gpu |wc -l"
    res = cmd(command)
    if res is not None and res == '1':
        use_gpu = True


def train_loader(type: str, batch_size):
    """
    数据加载
    :return:
    """
    # 数据加载
    st = time.time()
    paths = glob('{}/train/dataset-line/*/*.json'.format(path))  ##table line dataset label with labelme
    trainP, testP = train_test_split(paths, test_size=0.1)
    print('total:', len(paths), 'train:', len(trainP), 'test:', len(testP))
    dataloader = None
    if type == 'train':
        dataloader = MyDataset([512], trainP, 1)
    elif type == 'test':
        dataloader = MyDataset([512], testP, 1)
    datas = paddle.io.DataLoader(dataloader, batch_size=batch_size, shuffle=True)
    return datas


class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self, sizes: list, paths: list, linetype: int):
        """

        :param sizes: 训练尺度
        :param paths:
        :param linetype:
        """
        super(MyDataset, self).__init__()
        self.num_samples = len(paths)
        self.sizes = sizes
        self.paths = paths
        self.linetype = linetype
        sizes = self.sizes  ##多尺度训练
        self.data, self.lable = [], []
        for size in sizes:
            for path in self.paths:
                X = np.zeros((1, 3, size, size)).astype('float32')
                Y = np.zeros((1, 2, size, size)).astype('float32')
                img, lines, labelImg = get_img_label(path, size=(size, size), linetype=self.linetype)
                X = img
                Y = labelImg
                X = paddle.to_tensor(X, dtype="float32", )
                Y = paddle.to_tensor(Y, dtype="float32")
                self.data.append(X)
                self.lable.append(Y)

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """

        return self.data[index], self.lable[index]

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return self.num_samples


if __name__ == '__main__':
    # 模型权重存放位置
    # net = paddle.Model(Unet())
    use_gpu = False
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
    net = paddle.Model(model)
    EPOCH_NUM = 30
    BATCH_SIZE = 20
    # 为模型训练做准备，设置优化器，损失函数和精度计算方式 loss=paddle.nn.CrossEntropyLoss(soft_label=True),
    net.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=Unet().parameters()),
                loss=paddle.nn.functional.binary_cross_entropy,
                metrics=paddle.metric.Accuracy())
    # 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
    callback = paddle.callbacks.ModelCheckpoint(save_dir='{}/models/test.pd'.format(path))
    net.fit(train_data=train_loader("train", BATCH_SIZE),
            epochs=EPOCH_NUM,
            verbose=1,
            log_freq=10,
            save_freq=10,
            save_dir='{}/models/test'.format(path)
            )
    # paddle.save(model.state_dict(), 'mnist.pdparams')
    result = net.evaluate(train_loader('test', BATCH_SIZE), batch_size=BATCH_SIZE, log_freq=10)
    print(result)
