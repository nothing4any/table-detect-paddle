#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
table line detect
@author: chineseocr
"""

from conf.config_paddle import tableModeLinePath
from tools.utils import letterbox_image, get_table_line, adjust_lines, line_to_line
import numpy as np
import paddle
import cv2

def letterbox_image(image, size, fillValue=[128, 128, 128]):
    '''
    resize image with unchanged aspect ratio using padding
    '''
    image_h, image_w = image.shape[:2]
    w, h = size
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #cv2.imwrite('tmp/resized_image.png', resized_image[...,::-1])
    if fillValue is None:
        fillValue = [int(x.mean()) for x in cv2.split(np.array(image))]
    boxed_image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    boxed_image[:] = fillValue
    boxed_image[:new_h, :new_w, :] = resized_image
    # boxed_image.reshape((4, size[1], size[0], 3))
    #cv2.imwrite('tmp/boxed_image.png', boxed_image[..., ::-1])

    return boxed_image, new_w / image_w, new_h / image_h

class Unet(paddle.nn.Layer):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv0 = paddle.nn.Conv2D(in_channels=3, out_channels=16, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm0 = paddle.nn.BatchNorm2D(num_features=16, epsilon=9.999999747378752e-06)
        self.relu0 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv1 = paddle.nn.Conv2D(in_channels=16, out_channels=16, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm1 = paddle.nn.BatchNorm2D(num_features=16, epsilon=9.999999747378752e-06)
        self.relu1 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.pool0 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)

        self.conv2 = paddle.nn.Conv2D(in_channels=16, out_channels=32, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm2 = paddle.nn.BatchNorm2D(num_features=32, epsilon=9.999999747378752e-06)
        self.relu2 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv3 = paddle.nn.Conv2D(in_channels=32, out_channels=32, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm3 = paddle.nn.BatchNorm2D(num_features=32, epsilon=9.999999747378752e-06)
        self.relu3 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)

        self.conv4 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm4 = paddle.nn.BatchNorm2D(num_features=64, epsilon=9.999999747378752e-06)
        self.relu4 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv5 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm5 = paddle.nn.BatchNorm2D(num_features=64, epsilon=9.999999747378752e-06)
        self.relu5 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)

        self.conv6 = paddle.nn.Conv2D(in_channels=64, out_channels=128, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm6 = paddle.nn.BatchNorm2D(num_features=128, epsilon=9.999999747378752e-06)
        self.relu6 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv7 = paddle.nn.Conv2D(in_channels=128, out_channels=128, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm7 = paddle.nn.BatchNorm2D(num_features=128, epsilon=9.999999747378752e-06)
        self.relu7 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.pool3 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)

        self.conv8 = paddle.nn.Conv2D(in_channels=128, out_channels=256, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm8 = paddle.nn.BatchNorm2D(num_features=256, epsilon=9.999999747378752e-06)
        self.relu8 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv9 = paddle.nn.Conv2D(in_channels=256, out_channels=256, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm9 = paddle.nn.BatchNorm2D(num_features=256, epsilon=9.999999747378752e-06)
        self.relu9 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.pool4 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)

        self.conv10 = paddle.nn.Conv2D(in_channels=256, out_channels=512, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm10 = paddle.nn.BatchNorm2D(num_features=512, epsilon=9.999999747378752e-06)
        self.relu10 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv11 = paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm11 = paddle.nn.BatchNorm2D(num_features=512, epsilon=9.999999747378752e-06)
        self.relu11 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.pool5 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)

        self.conv12 = paddle.nn.Conv2D(in_channels=512, out_channels=1024, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm12 = paddle.nn.BatchNorm2D(num_features=1024, epsilon=9.999999747378752e-06)
        self.relu12 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv13 = paddle.nn.Conv2D(in_channels=1024, out_channels=1024, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm13 = paddle.nn.BatchNorm2D(num_features=1024, epsilon=9.999999747378752e-06)
        self.relu13 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv14 = paddle.nn.Conv2DTranspose(in_channels=1024, out_channels=1024, kernel_size=[4, 4], stride=2,
                                                padding=1, groups=1024, bias_attr=False)

        self.conv15 = paddle.nn.Conv2D(in_channels=1536, out_channels=512, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm14 = paddle.nn.BatchNorm2D(num_features=512, epsilon=9.999999747378752e-06)
        self.relu14 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv16 = paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm15 = paddle.nn.BatchNorm2D(num_features=512, epsilon=9.999999747378752e-06)
        self.relu15 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv17 = paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm16 = paddle.nn.BatchNorm2D(num_features=512, epsilon=9.999999747378752e-06)
        self.relu16 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv18 = paddle.nn.Conv2DTranspose(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=2,
                                                padding=1, groups=512, bias_attr=False)

        self.conv19 = paddle.nn.Conv2D(in_channels=768, out_channels=256, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm17 = paddle.nn.BatchNorm2D(num_features=256, epsilon=9.999999747378752e-06)
        self.relu17 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv20 = paddle.nn.Conv2D(in_channels=256, out_channels=256, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm18 = paddle.nn.BatchNorm2D(num_features=256, epsilon=9.999999747378752e-06)
        self.relu18 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv21 = paddle.nn.Conv2D(in_channels=256, out_channels=256, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm19 = paddle.nn.BatchNorm2D(num_features=256, epsilon=9.999999747378752e-06)
        self.relu19 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv22 = paddle.nn.Conv2DTranspose(in_channels=256, out_channels=256, kernel_size=[4, 4], stride=2,
                                                padding=1, groups=256, bias_attr=False)

        self.conv23 = paddle.nn.Conv2D(in_channels=384, out_channels=128, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm20 = paddle.nn.BatchNorm2D(num_features=128, epsilon=9.999999747378752e-06)
        self.relu20 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv24 = paddle.nn.Conv2D(in_channels=128, out_channels=128, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm21 = paddle.nn.BatchNorm2D(num_features=128, epsilon=9.999999747378752e-06)
        self.relu21 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv25 = paddle.nn.Conv2D(in_channels=128, out_channels=128, kernel_size=[3, 3], padding=1,
                                       bias_attr=False)
        self.batchnorm22 = paddle.nn.BatchNorm2D(num_features=128, epsilon=9.999999747378752e-06)
        self.relu22 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv26 = paddle.nn.Conv2DTranspose(in_channels=128, out_channels=128, kernel_size=[4, 4], stride=2,
                                                padding=1, groups=128, bias_attr=False)

        self.conv27 = paddle.nn.Conv2D(in_channels=192, out_channels=64, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm23 = paddle.nn.BatchNorm2D(num_features=64, epsilon=9.999999747378752e-06)
        self.relu23 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv28 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm24 = paddle.nn.BatchNorm2D(num_features=64, epsilon=9.999999747378752e-06)
        self.relu24 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv29 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm25 = paddle.nn.BatchNorm2D(num_features=64, epsilon=9.999999747378752e-06)
        self.relu25 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv30 = paddle.nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=[4, 4], stride=2,
                                                padding=1, groups=64, bias_attr=False)

        self.conv31 = paddle.nn.Conv2D(in_channels=96, out_channels=32, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm26 = paddle.nn.BatchNorm2D(num_features=32, epsilon=9.999999747378752e-06)
        self.relu26 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv32 = paddle.nn.Conv2D(in_channels=32, out_channels=32, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm27 = paddle.nn.BatchNorm2D(num_features=32, epsilon=9.999999747378752e-06)
        self.relu27 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv33 = paddle.nn.Conv2D(in_channels=32, out_channels=32, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm28 = paddle.nn.BatchNorm2D(num_features=32, epsilon=9.999999747378752e-06)
        self.relu28 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)
        self.conv34 = paddle.nn.Conv2DTranspose(in_channels=32, out_channels=32, kernel_size=[4, 4], stride=2,
                                                padding=1, groups=32, bias_attr=False)

        self.conv35 = paddle.nn.Conv2D(in_channels=48, out_channels=16, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm29 = paddle.nn.BatchNorm2D(num_features=16, epsilon=9.999999747378752e-06)
        self.relu29 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv36 = paddle.nn.Conv2D(in_channels=16, out_channels=16, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm30 = paddle.nn.BatchNorm2D(num_features=16, epsilon=9.999999747378752e-06)
        self.relu30 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv37 = paddle.nn.Conv2D(in_channels=16, out_channels=16, kernel_size=[3, 3], padding=1, bias_attr=False)
        self.batchnorm31 = paddle.nn.BatchNorm2D(num_features=16, epsilon=9.999999747378752e-06)
        self.relu31 = paddle.nn.LeakyReLU(negative_slope=0.10000000149011612)

        self.conv38 = paddle.nn.Conv2D(in_channels=16, out_channels=2, kernel_size=[1, 1])
        self.sigmoid0 = paddle.nn.layer.Sigmoid()

    def forward(self, data):
        # There are 1 inputs.
        # data: shape-[-1, 3, 512, 512], type-float32.
        conv2d = self.conv0(data)
        batch_normalizations = self.batchnorm0(conv2d)
        leaky_re_lu = self.relu0(batch_normalizations)
        conv2d_1 = self.conv1(leaky_re_lu)
        batch_normalization_1s = self.batchnorm1(conv2d_1)
        leaky_re_lu_1 = self.relu1(batch_normalization_1s)
        max_pooling2d = self.pool0(leaky_re_lu_1)
        conv2d_2 = self.conv2(max_pooling2d)
        batch_normalization_2s = self.batchnorm2(conv2d_2)
        leaky_re_lu_2 = self.relu2(batch_normalization_2s)
        conv2d_3 = self.conv3(leaky_re_lu_2)
        batch_normalization_3s = self.batchnorm3(conv2d_3)
        leaky_re_lu_3 = self.relu3(batch_normalization_3s)
        max_pooling2d_1 = self.pool1(leaky_re_lu_3)
        conv2d_4 = self.conv4(max_pooling2d_1)
        batch_normalization_4s = self.batchnorm4(conv2d_4)
        leaky_re_lu_4 = self.relu4(batch_normalization_4s)
        conv2d_5 = self.conv5(leaky_re_lu_4)
        batch_normalization_5s = self.batchnorm5(conv2d_5)
        leaky_re_lu_5 = self.relu5(batch_normalization_5s)
        max_pooling2d_2 = self.pool2(leaky_re_lu_5)
        conv2d_6 = self.conv6(max_pooling2d_2)
        batch_normalization_6s = self.batchnorm6(conv2d_6)
        leaky_re_lu_6 = self.relu6(batch_normalization_6s)
        conv2d_7 = self.conv7(leaky_re_lu_6)
        batch_normalization_7s = self.batchnorm7(conv2d_7)
        leaky_re_lu_7 = self.relu7(batch_normalization_7s)
        max_pooling2d_3 = self.pool3(leaky_re_lu_7)
        conv2d_8 = self.conv8(max_pooling2d_3)
        batch_normalization_8s = self.batchnorm8(conv2d_8)
        leaky_re_lu_8 = self.relu8(batch_normalization_8s)
        conv2d_9 = self.conv9(leaky_re_lu_8)
        batch_normalization_9s = self.batchnorm9(conv2d_9)
        leaky_re_lu_9 = self.relu9(batch_normalization_9s)
        max_pooling2d_4 = self.pool4(leaky_re_lu_9)
        conv2d_10 = self.conv10(max_pooling2d_4)
        batch_normalization_10s = self.batchnorm10(conv2d_10)
        leaky_re_lu_10 = self.relu10(batch_normalization_10s)
        conv2d_11 = self.conv11(leaky_re_lu_10)
        batch_normalization_11s = self.batchnorm11(conv2d_11)
        leaky_re_lu_11 = self.relu11(batch_normalization_11s)
        max_pooling2d_5 = self.pool5(leaky_re_lu_11)
        conv2d_12 = self.conv12(max_pooling2d_5)
        batch_normalization_12s = self.batchnorm12(conv2d_12)
        leaky_re_lu_12 = self.relu12(batch_normalization_12s)
        conv2d_13 = self.conv13(leaky_re_lu_12)
        batch_normalization_13s = self.batchnorm13(conv2d_13)
        leaky_re_lu_13 = self.relu13(batch_normalization_13s)
        up_sampling2d = self.conv14(leaky_re_lu_13)
        concatenate = paddle.concat(x=[leaky_re_lu_11, up_sampling2d], axis=1)
        conv2d_14 = self.conv15(concatenate)
        batch_normalization_14s = self.batchnorm14(conv2d_14)
        leaky_re_lu_14 = self.relu14(batch_normalization_14s)
        conv2d_15 = self.conv16(leaky_re_lu_14)
        batch_normalization_15s = self.batchnorm15(conv2d_15)
        leaky_re_lu_15 = self.relu15(batch_normalization_15s)
        conv2d_16 = self.conv17(leaky_re_lu_15)
        batch_normalization_16s = self.batchnorm16(conv2d_16)
        leaky_re_lu_16 = self.relu16(batch_normalization_16s)
        up_sampling2d_1 = self.conv18(leaky_re_lu_16)
        concatenate_1 = paddle.concat(x=[leaky_re_lu_9, up_sampling2d_1], axis=1)
        conv2d_17 = self.conv19(concatenate_1)
        batch_normalization_17s = self.batchnorm17(conv2d_17)
        leaky_re_lu_17 = self.relu17(batch_normalization_17s)
        conv2d_18 = self.conv20(leaky_re_lu_17)
        batch_normalization_18s = self.batchnorm18(conv2d_18)
        leaky_re_lu_18 = self.relu18(batch_normalization_18s)
        conv2d_19 = self.conv21(leaky_re_lu_18)
        batch_normalization_19s = self.batchnorm19(conv2d_19)
        leaky_re_lu_19 = self.relu19(batch_normalization_19s)
        up_sampling2d_2 = self.conv22(leaky_re_lu_19)
        concatenate_2 = paddle.concat(x=[leaky_re_lu_7, up_sampling2d_2], axis=1)
        conv2d_20 = self.conv23(concatenate_2)
        batch_normalization_20s = self.batchnorm20(conv2d_20)
        leaky_re_lu_20 = self.relu20(batch_normalization_20s)
        conv2d_21 = self.conv24(leaky_re_lu_20)
        batch_normalization_21s = self.batchnorm21(conv2d_21)
        leaky_re_lu_21 = self.relu21(batch_normalization_21s)
        conv2d_22 = self.conv25(leaky_re_lu_21)
        batch_normalization_22s = self.batchnorm22(conv2d_22)
        leaky_re_lu_22 = self.relu22(batch_normalization_22s)
        up_sampling2d_3 = self.conv26(leaky_re_lu_22)
        concatenate_3 = paddle.concat(x=[leaky_re_lu_5, up_sampling2d_3], axis=1)
        conv2d_23 = self.conv27(concatenate_3)
        batch_normalization_23s = self.batchnorm23(conv2d_23)
        leaky_re_lu_23 = self.relu23(batch_normalization_23s)
        conv2d_24 = self.conv28(leaky_re_lu_23)
        batch_normalization_24s = self.batchnorm24(conv2d_24)
        leaky_re_lu_24 = self.relu24(batch_normalization_24s)
        conv2d_25 = self.conv29(leaky_re_lu_24)
        batch_normalization_25s = self.batchnorm25(conv2d_25)
        leaky_re_lu_25 = self.relu25(batch_normalization_25s)
        up_sampling2d_4 = self.conv30(leaky_re_lu_25)
        concatenate_4 = paddle.concat(x=[leaky_re_lu_3, up_sampling2d_4], axis=1)
        conv2d_26 = self.conv31(concatenate_4)
        batch_normalization_26s = self.batchnorm26(conv2d_26)
        leaky_re_lu_26 = self.relu26(batch_normalization_26s)
        conv2d_27 = self.conv32(leaky_re_lu_26)
        batch_normalization_27s = self.batchnorm27(conv2d_27)
        leaky_re_lu_27 = self.relu27(batch_normalization_27s)
        conv2d_28 = self.conv33(leaky_re_lu_27)
        batch_normalization_28s = self.batchnorm28(conv2d_28)
        leaky_re_lu_28 = self.relu28(batch_normalization_28s)
        up_sampling2d_5 = self.conv34(leaky_re_lu_28)
        concatenate_5 = paddle.concat(x=[leaky_re_lu_1, up_sampling2d_5], axis=1)
        conv2d_29 = self.conv35(concatenate_5)
        batch_normalization_29s = self.batchnorm29(conv2d_29)
        leaky_re_lu_29 = self.relu29(batch_normalization_29s)
        conv2d_30 = self.conv36(leaky_re_lu_29)
        batch_normalization_30s = self.batchnorm30(conv2d_30)
        leaky_re_lu_30 = self.relu30(batch_normalization_30s)
        conv2d_31 = self.conv37(leaky_re_lu_30)
        batch_normalization_31s = self.batchnorm31(conv2d_31)
        leaky_re_lu_31 = self.relu31(batch_normalization_31s)
        conv2d_32 = self.conv38(leaky_re_lu_31)
        conv2d_32s = self.sigmoid0(conv2d_32)
        return conv2d_32s

paddle.disable_static()
params = paddle.load(tableModeLinePath)
model = Unet()
model.set_dict(params, use_structured_name=True)
model.eval()


def table_line(img, size=(512, 512), hprob=0.5, vprob=0.5, row=50, col=15, alph=20):
    sizew, sizeh = size
    inputBlob, fx, fy = letterbox_image(img[..., ::-1], (sizew, sizeh))
   
    input_img = inputBlob.reshape((sizew, sizeh, 3)) / 255.0
    input_img = input_img.astype(np.float32, copy = False)

    input_img = paddle.to_tensor(input_img)

    input_img = paddle.transpose(input_img, perm=[2, 0, 1])
    input_img = paddle.unsqueeze(input_img, axis=0)


    pred = model(input_img)
    result = pred[0]
    result = result.numpy()

    vpred = result[1,...] > vprob  ##竖线
    hpred = result[0,...] > hprob  ##横线
    vpred = vpred.astype(int)
    hpred = hpred.astype(int)

    colboxes = get_table_line(vpred, axis=1, lineW=col)
    rowboxes = get_table_line(hpred, axis=0, lineW=row)
    ccolbox = []
    crowlbox = []
    if len(rowboxes) > 0:
        rowboxes = np.array(rowboxes)
        rowboxes[:, [0, 2]] = rowboxes[:, [0, 2]] / fx
        rowboxes[:, [1, 3]] = rowboxes[:, [1, 3]] / fy
        xmin = rowboxes[:, [0, 2]].min()
        xmax = rowboxes[:, [0, 2]].max()
        ymin = rowboxes[:, [1, 3]].min()
        ymax = rowboxes[:, [1, 3]].max()
        ccolbox = [[xmin, ymin, xmin, ymax], [xmax, ymin, xmax, ymax]]
        rowboxes = rowboxes.tolist()

    if len(colboxes) > 0:
        colboxes = np.array(colboxes)
        colboxes[:, [0, 2]] = colboxes[:, [0, 2]] / fx
        colboxes[:, [1, 3]] = colboxes[:, [1, 3]] / fy

        xmin = colboxes[:, [0, 2]].min()
        xmax = colboxes[:, [0, 2]].max()
        ymin = colboxes[:, [1, 3]].min()
        ymax = colboxes[:, [1, 3]].max()
        colboxes = colboxes.tolist()
        crowlbox = [[xmin, ymin, xmax, ymin], [xmin, ymax, xmax, ymax]]

    rowboxes += crowlbox
    #colboxes += ccolbox

    rboxes_row_, rboxes_col_ = adjust_lines(rowboxes, colboxes, alph=alph)
    rowboxes += rboxes_row_
    colboxes += rboxes_col_
    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
        for j in range(ncol):
            rowboxes[i] = line_to_line(rowboxes[i], colboxes[j], 20)
            colboxes[j] = line_to_line(colboxes[j], rowboxes[i], 11)


    return rowboxes, colboxes



if __name__ == '__main__':
    import time

    p = 'img/test1.png'
    from tools.utils import draw_lines

    img = cv2.imread(p)
    t = time.time()
    rowboxes, colboxes = table_line(img[..., ::-1], size=(512, 512), hprob=0.5, vprob=0.6)
    img = draw_lines(img, rowboxes + colboxes, color=(255, 0, 0), lineW=2)

    print(time.time() - t, len(rowboxes), len(colboxes))
    cv2.imwrite('img/table-line.png', img)
    print("done")
