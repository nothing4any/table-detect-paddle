#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2022/1/31 下午12:48
@file: table_ceil_paddle
@author: nothing4any
"""

import os
import time

import cv2
import numpy as np
from flask import *

from table_build import tableBuid, to_excel
from table_detect import table_detect
from table_line_paddle import table_line
from tools.utils import draw_boxes
from tools.utils import minAreaRectbox, measure, eval_angle, draw_lines

class table:
    def __init__(self, img, tableSize=(416, 416), tableLineSize=(1024, 1024), isTableDetect=False, isToExcel=False):
        self.img = img
        self.tableSize = tableSize
        self.tableLineSize = tableLineSize
        self.isTableDetect = isTableDetect
        self.isToExcel = isToExcel
        self.img_degree()
        self.table_boxes_detect()  ##表格定位
        self.table_ceil()  ##表格单元格定位

        self.table_build()

    def img_degree(self):
        img, degree = eval_angle(self.img, angleRange=[-15, 15])
        self.img = img
        self.degree = degree

    def table_boxes_detect(self):
        h, w = self.img.shape[:2]

        if self.isTableDetect:
            boxes, adBoxes, scores = table_detect(self.img, sc=self.tableSize, thresh=0.5, NMSthresh=0.3)
            if len(boxes) == 0:
                boxes = [[0, 0, w, h]]
                adBoxes = [[0, 0, w, h]]
                scores = [0]
        else:
            boxes = [[0, 0, w, h]]
            adBoxes = [[0, 0, w, h]]
            scores = [0]

        self.boxes = boxes
        self.adBoxes = adBoxes
        self.scores = scores

    def table_ceil(self):
        ###表格单元格
        n = len(self.adBoxes)
        self.tableCeilBoxes = []
        self.childImgs = []
        for i in range(n):
            xmin, ymin, xmax, ymax = [int(x) for x in self.adBoxes[i]]

            childImg = self.img[ymin:ymax, xmin:xmax]
            rowboxes, colboxes = table_line(childImg[..., ::-1], size=self.tableLineSize, hprob=0.4, vprob=0.35)
            tmp = np.zeros(self.img.shape[:2], dtype='uint8')
            tmp = draw_lines(tmp, rowboxes + colboxes, color=255, lineW=2)
            cv2.imwrite('img/table-line1.png', tmp)
            labels = measure.label(tmp < 255, connectivity=2)  # 8连通区域标记
            regions = measure.regionprops(labels)
            ceilboxes = minAreaRectbox(regions, False, tmp.shape[1], tmp.shape[0], True, False)
            ceilboxes = np.array(ceilboxes)
            ceilboxes[:, [0, 2, 4, 6]] += xmin
            ceilboxes[:, [1, 3, 5, 7]] += ymin

            self.tableCeilBoxes.extend(ceilboxes)
            self.childImgs.append(childImg)

    def table_build(self):
        tablebuild = tableBuid(self.tableCeilBoxes)
        cor = tablebuild.cor
        for line in cor:
            line['text'] = 'table-test'  ##ocr
        if self.isToExcel:
            workbook = to_excel(cor, workbook=None)
        else:
            workbook = None
        self.res = cor
        self.workbook = workbook

    def table_ocr(self):
        """use ocr and match ceil"""
        h, w = self.img.shape[:2]
        self.crops = []
        boxes = self.tableCeilBoxes
        line = []
        self.lines = []
        self.enter_line = []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
            if (round(x1, 1) == 0.0 and round(y1, 1) == 0.0) or abs(y1-y4) > h*0.82:
                del boxes[index]
        if h < 1000:
            for index, box in enumerate(boxes):
                if len(line) == 0:
                    line.append(box.tolist())
                elif abs(round(box[1], 1) - round(boxes[index - 1][1], 1)) <= 20 and abs(round(box[3], 1) - round(boxes[index - 1][3],
                                                                                                     1)) <= 20:
                    line.append(box.tolist())
                elif len(line) != 0 and (abs(
                        round(box[1], 1) - round(boxes[index - 1][1], 1)) > 20 and abs(round(box[5], 1) - round(boxes[index - 1][5],
                                                                                                        1)) > 20):
                    line.sort(key=lambda x: x[0])
                    self.lines.append(line)
                    line = []
                    line.append(box.tolist())
                if index == len(boxes) - 1:
                    self.lines.append(line)
        elif h > 1000:
            for index, box in enumerate(boxes):
                x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
                if abs(y1 - y4) <= 20 or abs(x2 - x1) <= 20 \
                        or (abs(x1 - x2) > w - 95 and abs(y1 - y4) < 30) or x2 > w or y1 > h \
                        or (abs(x1 - x2) > w - 95 and abs(y1 - y4) > h - 30):
                    del boxes[index]
            for index, box in enumerate(boxes):
                if len(line) == 0:
                    line.append(box.tolist())
                elif abs(round(box[1], 1) - round(boxes[index - 1][1], 1)) <= 40 and abs(round(box[3], 1) - round(boxes[index - 1][3],
                                                                                                     1)) <= 40\
                        and abs(round(box[5], 1) - round(boxes[index - 1][5], 1)) <= 40:
                    line.append(box.tolist())
                elif len(line) != 0 and (abs(
                        round(box[1], 1) - round(boxes[index - 1][1], 1)) > 40 and abs(round(box[5], 1) - round(boxes[index - 1][5],
                                                                                                        1)) > 40):
                    line.sort(key=lambda x: x[0])
                    self.lines.append(line)
                    line = []
                    line.append(box.tolist())
                if index == len(boxes) - 1:
                    self.lines.append(line)
        for index, x in enumerate(self.lines):
            if len(self.enter_line) == 0:
                self.enter_line.append(len(x) - 1)
            else:
                self.enter_line.append(self.enter_line[index - 1] + len(x))

    def clear(self):
        self.res = None
        self.img = None
        self.enter_line = None
        self.lines = None
        self.boxes = None
        self.adBoxes = None
        self.workbook = None
        self.crops = None
        self.tableCeilBoxes = None
        self.childImgs = None
        self.scores = None
        self.degree = None



def resize_img(image, input_size=960):
        """
        resize img and limit the longest side of the image to input_size
        """
        img = np.array(image)
        im_shape = img.shape
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(input_size) / float(im_size_max)
        if im_size_max < 1000:
            img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
            return img
        else:
            return image


def base64_to_cv2(b64str):
        import base64
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.fromstring(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data

app = Flask(__name__)


@app.route('/bigData/dossierRecognition/tableDetect/tableCeil', methods=["POST"])
def main():
    if request.method == 'POST':
        imgStreamInfo = request.json
        imgpath = imgStreamInfo["img"]
        if not imgpath.startswith(r'/') or len(imgpath)>100:
            img = base64_to_cv2(imgpath)
        else:
            img = cv2.imread(imgpath)
        img = resize_img(img, 2592)
        isTableDetect = imgStreamInfo["isTableDetect"]
        tableSize = imgStreamInfo["tableSize"]
        tableLineSize = imgStreamInfo["tableLineSize"]
        isToExcel = imgStreamInfo["isToExcel"]
        tableSize = [int(x) for x in tableSize.split(',')]
        tableLineSize = [int(x) for x in tableLineSize.split(',')]

        t = time.time()
        tableDetect = table(img, tableSize=tableSize,
                            tableLineSize=tableLineSize
                            )
        # tableCeilBoxes = tableDetect.tableCeilBoxes
        # tableJson = tableDetect.res
        # workbook = tableDetect.workbook
        img = tableDetect.img
        tmp = np.zeros_like(img)
        img = draw_boxes(tmp, tableDetect.tableCeilBoxes, color=(255, 255, 255))
        print(time.time() - t)
        tody_date = str(time.strftime("%Y%m%d", time.localtime()))
        if os.path.exists(
                "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")) + "/img/{}/".format(tody_date)):
            pass
        else:
            os.makedirs("/".join(os.path.dirname(os.path.abspath(__file__)).split("/")) + "/img/{}/".format(tody_date))
        pngP = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")) + r"/img/{}/".format(tody_date) + \
               imgpath[4:9] + 'ceil.png'
        print(pngP)
        cv2.imwrite(pngP, img)
        # if workbook is not None:
        #     workbook.save(os.path.splitext(img)[0] + '.xlsx')
        tableDetect.table_ocr()
        enter_line = tableDetect.enter_line
        lines = tableDetect.lines
        tableDetect.clear()
        return jsonify({"enter_line": enter_line, "lines": lines})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)