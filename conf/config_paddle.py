#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020

@author: chineseocr
"""
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
path = "/".join(dir_path.split("/")[:-1])
tableModelDetectPath = '{}/models/table-detect.weights'.format(path)
tableModeLinePath = '/opt/programs/python/table-detect-paddle/models/model.pdparams'
# tableModeLinePath = '/opt/programs/python/table-detect-paddle/models/test/0.pdparams'
