#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2022/2/8 下午8:22
@file: tilt_correction
@author: nothing4any
原文 https://blog.csdn.net/feilong_csdn/article/details/81586322
"""

import math

import cv2
import numpy as np
from scipy import ndimage


def run(input):
    """
    图像倾斜矫正
    :param input: image path
    :return: ndarray
    """
    img = cv2.imread(input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 霍夫变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(img, rotate_angle)
    return rotate_img

if __name__ == '__main__':
    path = '/opt/programs/python/table-detect-paddle/img/SKM_C364e21101611040_0021.jpg'
    rotated = run(path)
    cv2.imwrite("output.jpg", rotated)