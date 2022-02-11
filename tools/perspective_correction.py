#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2022/2/9 上午8:11
@file: perspective_correction
@author: nothing4any
原文：https://blog.csdn.net/qq_41821678/article/details/106851010?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&utm_relevant_index=2
"""
import cv2
import numpy as np


# # 预处理，高斯滤波（用处不大），4次开操作
# # 过滤轮廓唯一
def contour_demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    ref, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    return contours

# 计算透视变换参数矩阵
def cal_perspective_params(img, points):
    # 设置偏移点。如果设置为(0,0),表示透视结果只显示变换的部分（也就是画框的部分）
    offset_x = 100
    offset_y = 100
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(points)
    # 透视变换的四个点
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y],
                      [offset_x, img_size[1] - offset_y], [img_size[0] - offset_x, img_size[1] - offset_y]])
    # 透视矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # 透视逆矩阵
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    return M, M_inverse

# 透视变换
def img_perspect_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size)

def draw_line(img,p1,p2,p3,p4):
    points = [list(p1), list(p2), list(p3), list(p4)]
    # 画线
    img = cv2.line(img, p1, p2, (0, 0, 255), 3)
    img = cv2.line(img, p2, p4, (0, 0, 255), 3)
    img = cv2.line(img, p4, p3, (0, 0, 255), 3)
    img = cv2.line(img, p3, p1, (0, 0, 255), 3)
    return points, img

def run(input):
    img = cv2.imread(input)
    contours = contour_demo(img)
    contour = contours[0]
    # 选取四个点，分别是左上、右上、左下、右下
    img_copy = img.copy()
    # 使用approxPolyDP，将轮廓转换为直线，22为精度（越高越低），TRUE为闭合
    approx = cv2.approxPolyDP(contour, 22, True)
    n = []
    # 生产四个角的坐标点
    for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
        n.append((x, y))
    points = [list(n[1]), list(n[0]), list(n[2]), list(n[3])]
    M, M_inverse = cal_perspective_params(img, points)
    trasform_img = img_perspect_transform(img, M)

    return trasform_img
if __name__ == '__main__':
    path = '/home/solo/桌面/微信图片_20220209082025.jpg'
    trasform_img = run(path)
    cv2.imwrite('test02.png',trasform_img)







