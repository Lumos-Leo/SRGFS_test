#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :ImageUtils.py
@说明        :
@时间        :2021/05/29 15:47:42
@作者        :Oasis
@版本        :1.0
'''


import random
import re

import cv2 as cv
import numpy
import torch
import skimage.color as color
import skimage.io as io
import skimage.transform as transform

from PIL import Image
from torch import uint8
from torch.utils import data


# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(numpy.ascontiguousarray(img)).permute(2, 0, 1).float()
        
def imgResize(img, shape):
    return transform.resize(img, shape, order=3)

def imgRgb2Ycbcr(img):
    # b, g, r = cv.split(img)
    # y = 0.257*r + 0.504*g + 0.098*b + 16
    # cb = -0.148*r - 0.291*g + 0.439*b + 128
    # cr = 0.439*r - 0.368*g - 0.071*b + 128
    # return cv.merge([y, cb, cr])
    return color.rgb2ycbcr(img)

def imgRgb2Y(img):
    if len(img.shape) == 2:
        return numpy.expand_dims(img, 2)
    img = color.rgb2ycbcr(img)
    return img[:,:,0:1]

def imgYcbcr2Rgb(img):
    # y, cb, cr = cv.split(img)
    # r = 1.164*(y-16) + 1.596*(cr-128)
    # g = 1.164*(y-16) - 0.813*(cr-128) - 0.392*(cb-128)
    # b = 1.164*(y-16) + 2.017*(cb-128) 
    # return cv.merge([r, g, b])
    return color.ycbcr2rgb(img)

def imgRead(path, type_='', color_space=''):
    img = io.imread(path)
    if color_space == "ycrcb":
        img = imgRgb2Ycbcr(img)
        y, cb, cr = cv.split(img)
        img = y
    if type_ == "pt":
        img = numpy.expand_dims(img, 0)
    return img

def imgSave(img, path, data_range='255'):
    if data_range == '255':
        io.imsave(path, img.numpy().astype(numpy.uint8))
    elif data_range == '1':
        img *= 255
        io.imsave(path, img.numpy().astype(numpy.uint8))
    else:
        raise 'error data range...'

def imgRandomCrop(img1, img2, size, scale):
    # 数据预处理，随机裁剪crop_size区域
    h1, w1, c1 = img1.shape
    th, tw = size
    if w1 == tw and h1 == th:
        out1 = img1
    else:
        x1 = random.randint(0, w1 - tw)
        y1 = random.randint(0, h1 - th)
        out1 = img1[y1:y1 + th, x1:x1 + tw, :]
        out2 = img2[y1*scale:y1*scale+th*scale, x1*scale:x1*scale+tw*scale, :]

    out1, out2 = augment((out1,out2))
    return out1, out2

def imgCrop(img1, size):
    # 数据预处理，随机裁剪crop_size区域
    h1, w1, c1 = img1.shape
    th, tw = size
    if w1 == tw and h1 == th:
        out1 = img1
    else:
        x1 = random.randint(0, w1 - tw)
        y1 = random.randint(0, h1 - th)
        out1 = img1[y1:y1 + th, x1:x1 + tw, :]

    return out1

def features_save(features):
    for layer,feature in enumerate(features):
        print(feature.shape)
        feature = feature.clamp(0.0, 1.0).cpu().numpy()
        for channel in range(feature.shape[1]):
            io.imsave('./results/features/layer{}_feature{}.png'.format(layer+1, channel+1), feature[0,channel,:,:]*255.)

def augment(args, hflip=True, rot=True):
    hflip = hflip and random.random() > 0.5
    vflip = rot and random.random() > 0.5
    rot90 = rot and random.random() > 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img
    return [_augment(img) for img in args]

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        img = img
    elif mode == 1:
        img = numpy.flipud(numpy.rot90(img))
    elif mode == 2:
        img = numpy.flipud(img)
    elif mode == 3:
        img = numpy.rot90(img, k=3)
    elif mode == 4:
        img = numpy.flipud(numpy.rot90(img, k=2))
    elif mode == 5:
        img = numpy.rot90(img)
    elif mode == 6:
        img = numpy.rot90(img, k=2)
    elif mode == 7:
        img = numpy.flipud(numpy.rot90(img, k=3))
    return img