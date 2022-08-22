#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :CommonUtils.py
@说明        :
@时间        :2021/05/29 15:47:14
@作者        :Oasis
@版本        :1.0
'''
import os
import time
import sys
import h5py
import numpy as np
import skimage.io as io
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.distributed as dist
import shutil

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.utils.data import DataLoader
from Utils.buildDataset import DIV2KCropTrainDataset, DIV2KCropValidDataset
from torch.utils.data.distributed import DistributedSampler
from imageUtils.ImageUtils import *
from models.maskunit import MaskUnit
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def collapse_Skip(weight_tmp,new_bias,f3,f3b):
    weight3 = f3
    bias3 = f3b

    # weight merge
    weight_tmp_ = weight_tmp.view(weight_tmp.size(0), weight_tmp.size(1),  weight_tmp.size(2) * weight_tmp.size(3))
    weight3_ = weight3.view(weight3.size(0), weight3.size(1) * weight3.size(2) * weight3.size(3))

    new_weight_ = torch.Tensor(weight3.size(0), weight_tmp.size(1), weight_tmp.size(2)*weight_tmp.size(3)).to('cuda')
    for i in range(weight_tmp.size(1)):
        tmp = weight_tmp_[:, i, :].view(weight_tmp.size(0),  weight_tmp.size(2) * weight_tmp.size(3))
        new_weight_[:, i, :] = torch.matmul(weight3_, tmp)
    weight_combi = new_weight_.view(weight3.size(0), weight_tmp.size(1),  weight_tmp.size(2), weight_tmp.size(3))


    if new_bias is not None and bias3 is not None:
        new_bias = torch.matmul(weight3_, new_bias) + bias3  # with bias
    elif new_bias is None:
        new_bias = bias3  #without Bias
    else:
        new_bias = None

    bia1_combi = new_bias
    weight_collapse = weight_combi
    return weight_collapse, bia1_combi


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

# --------------------------------------------
# imresize for numpy image [0, 1]
# --------------------------------------------
def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = torch.from_numpy(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()

def compute_ck(s_1, s_2):
    """
    Compute weight from 2 conv layers, whose kernel size larger than 3*3
    After derivation, F.conv_transpose2d can be used to compute weight of original conv layer
    :param s_1: 3*3 or larger conv layer
    :param s_2: 3*3 or larger conv layer
    :return: new weight and bias
    """
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    w_s_2_tmp = w_s_2.view(w_s_2.size(0), w_s_2.size(1), w_s_2.size(2) * w_s_2.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, b_s_1) + b_s_2
    elif b_s_1 is None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = None

    new_weight = F.conv_transpose2d(w_s_2, w_s_1)

    return {'weight': new_weight, 'bias': new_bias}


def expandDataset(hr_path_base, lr_path_base, valid_hr_path, valid_lr_path, cnt, size, scale, type_ori, dataset, logger, rank, base_path, ablation=False):
    if rank == 0:
        logger.logger.info('make test dataset **{}**'.format(dataset))
    outdir = os.path.join(base_path, 'benchmark', 'hdf5', dataset, 'X{}'.format(scale))
    outpath = os.path.join(outdir, 'test_database.hdf5')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    h5_file = h5py.File(outpath, 'w')

    lr_groups = h5_file.create_group('lr')
    hr_groups = h5_file.create_group('hr')

    for i,img_path in enumerate(os.listdir(hr_path_base)):
        if img_path.endswith(".png"):
            if rank == 0:
                logger.logger.info('process -> {}'.format(img_path))
            hr_path = os.path.join(hr_path_base, img_path)
            lr_path = os.path.join(lr_path_base, '{}x{}.png'.format(img_path.split('.')[0], scale))

            hr = io.imread(hr_path)
            lr = io.imread(lr_path)

            hr = imgRgb2Y(hr)
            lr = imgRgb2Y(lr)


            lr_groups.create_dataset(str(i), data=lr)
            hr_groups.create_dataset(str(i), data=hr)

    h5_file.close()

def preprocess(path, build_type, args, test=False):
    
    if build_type == 'train':
        train_sampler = None
        if torch.cuda.device_count() > 1 and not test:
            # multi-GPUs
            train_sampler = DistributedSampler(DIV2KCropTrainDataset(path), shuffle=True)
            data_loader = DataLoader(DIV2KCropTrainDataset(path), batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=train_sampler, num_workers=args.num_works)
        else:
            # default
            data_loader = DataLoader(DIV2KCropTrainDataset(path), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_works)
        return data_loader, train_sampler
    elif build_type == 'valid':
        valid_sampler = None
        data_loader = DataLoader(DIV2KCropValidDataset(path), batch_size=1, shuffle=False, pin_memory=True)
        return data_loader, valid_sampler
    else:
        test_sampler = None
        data_loader = DataLoader(DIV2KCropValidDataset(path), batch_size=1, shuffle=False, pin_memory=True)
        return data_loader, test_sampler

def computer_psnr(pre_y, label_y, scale, data_range=1.):
    pre_y = pre_y[:,:,scale:-scale, scale:-scale]
    label_y = label_y[:,:,scale:-scale, scale:-scale]
    b1,c1,h1,w1 = pre_y.shape
    b2,c2,h2,w2 = label_y.shape
    min_h = min(h1,h2)
    min_w = min(w1,w2)
    pre_y = pre_y[:,:,:min_h, :min_w]
    label_y = label_y[:,:,:min_h, :min_w]

    # mse = np.mean((pre_y - label_y)**2)
    # psnr = 10 * np.log10(data_range**2/mse)
    return calc_psnr(pre_y, label_y, data_range)

def calc_psnr(sr, hr, data_range):
    diff = (sr - hr) / data_range
    mse  = diff.pow(2).mean()
    psnr = -10 * torch.log10(mse)                    
    return psnr

def calc_ssim(sr, hr, data_range):
    # def ssim(
    #     X,
    #     Y,
    #     data_range=255,
    #     size_average=True,
    #     win_size=11,
    #     win_sigma=1.5,
    #     win=None,
    #     K=(0.01, 0.03),
    #     nonnegative_ssim=False,
    # )
    b1,c1,h1,w1 = sr.shape
    b2,c2,h2,w2 = hr.shape
    min_h = min(h1,h2)
    min_w = min(w1,w2)
    sr = sr[:,:,:min_h, :min_w]
    hr = hr[:,:,:min_h, :min_w]
    ssim_val = ssim(sr, hr, data_range, size_average=True)
    return ssim_val

def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min
    t_max=t_max
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def collapse_block(layer, collapse_weight, collapse_bias, nameIn='', residual=False, mode='collapse'):
    out_dict = {}
    cnt = 0
    for name, layer in layer.named_children():
        if isinstance(layer, nn.PReLU):
            collapse_weight[nameIn+'_'+'prelu'] = layer.weight.data
            continue
        weight1 = layer.weight.data
        bias1 = layer.bias

        out_dict['weight'+str(cnt)] = weight1
        out_dict['bias'+str(cnt)] = bias1
        cnt += 1

    if mode == 'collapse':
        # compute_ck()
        weight1 = out_dict["weight"+str(0)]
        bias1 = out_dict["bias"+str(0)]

        weight2 = out_dict["weight"+str(1)]
        bias2 = out_dict["bias"+str(1)]

        # weight3 = out_dict["weight"+str(2)]
        # bias3 = out_dict["bias"+str(2)]
        
        weight, bias = collapse_Skip(weight1,bias1,weight2,bias2)

        # weight, bias = collapse_CollapseLayer(weight1,bias1,weight2,bias2,weight3,bias3)
        collapse_weight[nameIn+'_weight_comb'] = weight
        collapse_bias[nameIn+'_bias_comb'] = bias
    else:
        weight1 = out_dict["weight"+str(0)]
        bias1 = out_dict["bias"+str(0)]

        weight2 = out_dict["weight"+str(1)]
        bias2 = out_dict["bias"+str(1)]
        # weight merge
        weight_combi, bias = collapse_Skip(weight1,bias1,weight2,bias2)
        if residual:
            # residual merge
            outDims, kernel_size = weight_combi.shape[0],weight_combi.shape[3]
            weight_residual = torch.zeros(weight_combi.shape).cuda()
            if kernel_size == 3:
                idx = 1
            if kernel_size == 5:
                idx = 2
            for i in range(outDims):
                weight_residual[i,i,idx,idx] = 1

            # residual combi
            weight_collapse = weight_residual + weight_combi
        else:
            weight_collapse = weight_combi

        collapse_weight[nameIn+'_weight_comb'] = weight_collapse
        

def collapse_imply(layer, collapse_weight, collapse_bias, nameIn='',skip=False):
    for name, layer in layer.named_children():
        if isinstance(layer, nn.Conv2d) and layer.bias is None:
            layer.weight.data = collapse_weight[nameIn+'_weight_comb']
            continue
        elif isinstance(layer, nn.PReLU):
            layer.weight.data = collapse_weight[nameIn+'_prelu']
            continue
        elif isinstance(layer, MaskUnit):
            for name, layer in layer.named_children():
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data = collapse_weight[nameIn+'_mask_weight']
                    layer.bias.data = collapse_bias[nameIn+'_mask_bias']
            continue
        else:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = collapse_weight[nameIn+'_weight_comb']
                if collapse_bias[nameIn+'_bias_comb'] is not None:
                    layer.bias.data = collapse_bias[nameIn+'_bias_comb']


def space_to_depth(in_tensor, down_scale):
    n, c, h, w = in_tensor.size()
    unfolded_x = torch.nn.functional.unfold(in_tensor, down_scale, stride=down_scale)
    return unfolded_x.view(n, c * down_scale ** 2, h // down_scale, w // down_scale)

def getMask_simple(x,scale,training):
    n, c, h, w = x.shape
    x_down = F.interpolate(x, (h//scale, w//scale), mode='bicubic', align_corners=False)
    x_up = F.interpolate(x_down, (h, w), mode='bicubic', align_corners=False)
    img_mae = torch.abs(x - x_up)
    img_mae = img_mae.view(n,c,-1)
    img_mae = torch.mean(img_mae, 1, keepdim=True)
    img_median = torch.median(img_mae, dim=2, keepdim=True)
    mask = (img_mae > img_median[0]).view(n,1,h,w)
    return mask

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def generate_idx(a,ori):
    A = torch.arange(3).view(-1, 1, 1)
    mask_indices = torch.nonzero(a.squeeze())

    # indices: dense to sparse (1x1)
    h_idx_1x1 = mask_indices[:, 0]
    w_idx_1x1 = mask_indices[:, 1]

    # indices: dense to sparse (3x3)
    mask_indices_repeat = mask_indices.unsqueeze(0).repeat([3, 1, 1]) + A

    h_idx_3x3 = mask_indices_repeat[..., 0].repeat(1, 3).view(-1)
    w_idx_3x3 = mask_indices_repeat[..., 1].repeat(3, 1).view(-1)

    # indices: sparse to sparse (3x3)
    indices = torch.arange(float(mask_indices.size(0))).view(1, -1).to(a.device) + 1
    a[0, 0, h_idx_1x1, w_idx_1x1] = indices

    idx_s2s = F.pad(a, [1, 1, 1, 1])[0, :, h_idx_3x3, w_idx_3x3].view(9, -1).long()
    pick = F.pad(ori, [1, 1, 1, 1])[0, :, h_idx_3x3, w_idx_3x3].view(9 * ori.size(1), -1)
    return pick,h_idx_1x1,w_idx_1x1

def sparse_conv(ori, pick, k, h_idx_1x1, w_idx_1x1):
    conv3x3 = nn.Conv2d(1,3,k,padding=k//2)
    kernel = conv3x3.weight.data.view(3, -1)
    sparse_out = torch.mm(kernel, pick)
    ori = ori.repeat((1,3,1,1))
    ori[0, :, h_idx_1x1, w_idx_1x1] = sparse_out
    return ori


import numpy as np
from PIL import Image 
from scipy.signal import convolve2d
 
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
 
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
 
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))
 
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
 
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
 
    return np.mean(np.mean(ssim_map))

def collapse_2layer(f1,f1b,f2,f2b):
    weight_tmp = f1
    new_bias = f1b

    weight3 = f2
    bias3 = f2b

    weight_tmp_ = weight_tmp.view(weight_tmp.size(0), weight_tmp.size(1),  weight_tmp.size(2) * weight_tmp.size(3))
    weight3_ = weight3.view(weight3.size(0), weight3.size(1) * weight3.size(2) * weight3.size(3))

    new_weight_ = torch.Tensor(weight3.size(0), weight_tmp.size(1), weight_tmp.size(2)*weight_tmp.size(3)).to('cuda')
    for i in range(weight_tmp.size(1)):
        tmp = weight_tmp_[:, i, :].view(weight_tmp.size(0),  weight_tmp.size(2) * weight_tmp.size(3))
        new_weight_[:, i, :] = torch.matmul(weight3_, tmp)
    weight_combi = new_weight_.view(weight3.size(0), weight_tmp.size(1),  weight_tmp.size(2), weight_tmp.size(3))


    # weight_combi = torch.matmul(weight3_tmp, weight_tmp_tmp).view((weight3.shape[0],weight_tmp.shape[1],weight_tmp.shape[2],weight_tmp.shape[3]))
    bia1_combi = None
    # bias merge
    if new_bias is not None:
        bia1_combi = torch.matmul(weight3.view((weight3.shape[0], weight3.shape[1], -1)).sum(2), new_bias) + bias3
    weight_collapse = weight_combi
    return weight_collapse, bia1_combi

def collapse_CollapseLayer(f1,f1b,f2,f2b,f3,f3b):
    weight1 = f1
    bias1 = f1b

    weight2 = f2
    bias2 = f2b

    weight3 = f3
    bias3 = f3b

    w_s_2_tmp = weight2.view(weight2.size(0), weight2.size(1), weight2.size(2) * weight2.size(3))

    if bias1 is not None and bias2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, bias1) + bias2
    elif bias1 is None:
        new_bias = bias2  #without Bias
    else:
        new_bias = None

    weight_tmp = F.conv_transpose2d(weight2, weight1)

    # weight merge
    # weight_tmp = weight_tmp.view((weight_tmp.shape[0], -1))
    # weight3 = weight3.view((weight3.shape[0], -1))
    weight_tmp_ = weight_tmp.view(weight_tmp.size(0), weight_tmp.size(1),  weight_tmp.size(2) * weight_tmp.size(3))
    weight3_ = weight3.view(weight3.size(0), weight3.size(1) * weight3.size(2) * weight3.size(3))

    new_weight_ = torch.Tensor(weight3.size(0), weight_tmp.size(1), weight_tmp.size(2)*weight_tmp.size(3)).to('cuda')
    for i in range(weight_tmp.size(1)):
        tmp = weight_tmp_[:, i, :].view(weight_tmp.size(0),  weight_tmp.size(2) * weight_tmp.size(3))
        new_weight_[:, i, :] = torch.matmul(weight3_, tmp)
    weight_combi = new_weight_.view(weight3.size(0), weight_tmp.size(1),  weight_tmp.size(2), weight_tmp.size(3))


    # weight_combi = torch.matmul(weight3_tmp, weight_tmp_tmp).view((weight3.shape[0],weight_tmp.shape[1],weight_tmp.shape[2],weight_tmp.shape[3]))
    bia1_combi = None
    # bias merge
    if bias1 is not None:
        bia1_combi = torch.matmul(weight3.view((weight3.shape[0], weight3.shape[1], -1)).sum(2), new_bias) + bias3
    weight_collapse = weight_combi
    return weight_collapse, bia1_combi

def collapse_Skip(weight_tmp,new_bias,f3,f3b, residual=False):

    weight3 = f3
    bias3 = f3b

    # weight merge
    weight_tmp_ = weight_tmp.view(weight_tmp.size(0), weight_tmp.size(1),  weight_tmp.size(2) * weight_tmp.size(3))
    weight3_ = weight3.view(weight3.size(0), weight3.size(1) * weight3.size(2) * weight3.size(3))

    new_weight_ = torch.Tensor(weight3.size(0), weight_tmp.size(1), weight_tmp.size(2)*weight_tmp.size(3)).to('cuda')
    for i in range(weight_tmp.size(1)):
        tmp = weight_tmp_[:, i, :].view(weight_tmp.size(0),  weight_tmp.size(2) * weight_tmp.size(3))
        new_weight_[:, i, :] = torch.matmul(weight3_, tmp)
    weight_combi = new_weight_.view(weight3.size(0), weight_tmp.size(1),  weight_tmp.size(2), weight_tmp.size(3))


    if new_bias is not None and bias3 is not None:
        new_bias = torch.matmul(weight3_, new_bias) + bias3  # with bias
    elif new_bias is None:
        new_bias = bias3  #without Bias
    else:
        new_bias = None


    # weight_combi = torch.matmul(weight3_tmp, weight_tmp_tmp).view((weight3.shape[0],weight_tmp.shape[1],weight_tmp.shape[2],weight_tmp.shape[3]))
    bia1_combi = new_bias
    weight_collapse = weight_combi

    if residual:
        # residual merge
        outDims, kernel_size = weight_combi.shape[0],weight_combi.shape[3]
        weight_residual = torch.zeros(weight_combi.shape).cuda()
        if kernel_size == 3:
            idx = 1
        if kernel_size == 5:
            idx = 2
        for i in range(outDims):
            weight_residual[i,i,idx,idx] = 1

        # residual combi
        weight_collapse = weight_residual + weight_combi
    else:
        weight_collapse = weight_combi

    return weight_collapse, bia1_combi

def time_test(net, input, device, isMeta=False, meta=None, prior=None):
    if device == 'cuda':
        if isMeta:
            torch.cuda.synchronize()
            tik = time.time()
            pre_y,_ = net((input, meta, prior))
            torch.cuda.synchronize()
            tok = time.time() - tik
        else:
            torch.cuda.synchronize()
            tik = time.time()
            pre_y = net(input)
            torch.cuda.synchronize()
            tok = time.time() - tik
    else :
        if isMeta:
            tik = time.time()
            pre_y,_ = net((input, meta, prior))
            tok = time.time() - tik
        else:
            tik = time.time()
            pre_y = net(input)
            tok = time.time() - tik
    return pre_y,tok

def remove_prefix(state_dict_):
    state_dict = {}
    for key in state_dict_:
        if key.startswith('module') and not key.startswith('module_list'):
            state_dict[key[7:]] = state_dict_[key]
        else:
            state_dict[key] = state_dict_[key]
    return state_dict

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

    
 
if __name__ == "__main__":
    im1 = Image.open("1.png")
    im2 = Image.open("2.png")
 
    print(compute_ssim(np.array(im1),np.array(im2)))
    
