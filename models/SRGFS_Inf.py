#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :sesr.py
@说明        :
@时间        :2021/05/29 15:47:36
@作者        :Oasis
@版本        :1.0
'''
import sys
sys.path.append('./')

import math
import torch 
import torch.nn as nn
from models.maskunit import MaskUnit

def apply_mask(x, mask, reverse=False):
    mask_hard = mask.hard
    assert mask_hard.shape[0] == x.shape[0]
    assert mask_hard.shape[2:4] == x.shape[2:4], (mask_hard.shape, x.shape)
    if reverse:
        return (mask_hard <= 0).float().expand_as(x) * x
    else:
        return mask_hard.float().expand_as(x) * x

class FGSM_Inf(nn.Module):
    def __init__(self, f):
        super(FGSM_Inf, self).__init__()
        self.f = f

        self.fusion = nn.Sequential(
            nn.Conv2d(f, 1, 1), 
            nn.ReLU(inplace=True)
        )
        self.mask_maker = nn.Sequential(
            nn.Conv2d(1,16,3,1,1),
            nn.PReLU(num_parameters=16)
        )
        self.mask = MaskUnit()
        
    def forward(self, features, meta):
        self.h, self.w = features.shape[2], features.shape[3]
        sub_feas = self.fusion(features)
        soft = self.mask_maker(sub_feas)
        mask = self.mask(soft, meta, tail=True)
        return mask

    def macs(self):
        macs = 0
        h, w = self.h, self.w
        macs += 1 * self.f * 1 * (h * w)  
        macs += 9 * 1 * self.f * (h * w)
        
        return macs
    
    def params(self):
        params = 0
        params += 1 * 1 * 1 * self.f
        params += 3 * 3 * 1 * self.f
        return params

class Sparse_Conv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask, ori, conv3x3):
        self.mask = mask
        self.h, self.w = ori.shape[2], ori.shape[3]
        self.c_in, self.c_out = conv3x3.in_channels, conv3x3.out_channels
        ori = apply_mask(ori, mask)
        out = conv3x3(ori)
        return out

    def macs(self):
        macs = 0
        macs += 9 * self.c_out *  self.c_in * self.h * self.w * (self.mask.hard.sum()/(self.c_in * self.h * self.w))
        # macs += 9 * self.c_out *  self.c_in * self.h * self.w * 0.5
        return macs

    def params(self):
        params = 0
        return params

class GFFS_Inf(nn.Module):
    def __init__(self, f, s):
        super(GFFS_Inf, self).__init__()
        self.f = f
        self.s = s
        self.conv = nn.Conv2d(f, int(f*s), 3, 1, 1)
        self.conv_skip = nn.Conv2d(f, int(f*s), 1)
        self.sparse_conv = Sparse_Conv()
        self.depth_wise = nn.Conv2d(int(f*s), int(f*(1-s)), 3, 1, 1, groups=int(f*s))
        self.fgsm = FGSM_Inf(f)
        self.ca = CAModule_Inf(f, f, 16)
        self.prelu = nn.PReLU(num_parameters=f)

    def forward(self, input):
        x, meta = input
        self.h, self.w = x.shape[2], x.shape[3]
        mask = self.fgsm(x, meta)

        # transport skip connection
        identity = self.conv_skip(x)

        # get specific layers
        x = self.sparse_conv(mask, x, self.conv)
        x = x + identity
        
        # activate ghost layers
        ghost = self.depth_wise(x)

        # concat
        out = torch.cat([x, ghost], dim=1)
        out = self.ca(out)
        out = self.prelu(out)
        
        return out, meta

    def macs(self):
        macs = 0
        # conv_11_skip
        macs += 1 * self.f * int(self.f*self.s) * (self.h * self.w)
        macs += self.fgsm.macs()
        macs += self.sparse_conv.macs()
        macs += 9 * int(self.f*(1 - self.s)) * (self.h * self.w)
        macs += self.ca.macs()

        return macs
        
    def params(self):
        params = 0
        params += 1 * 1 * self.f * int(self.f*self.s)
        params += self.fgsm.params()
        params += 3 * 3 * self.f * int(self.f*self.s)
        params += 3 * 3 * int(self.f*(1 - self.s))
        params += self.ca.params()
        return params



class CAModule_Inf(nn.Module):
    def __init__(self, channel, planes, reduction):
        super().__init__()
        self.channel = channel
        self.planes = planes
        self.reduction = reduction
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, planes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x*self.ca(x)

    def macs(self):
        macs = 0
        # conv 1
        macs += 1 * self.channel * self.channel//self.reduction * (1 * 1)
        # conv 2
        macs += 1 * self.channel//self.reduction * self.planes * (1 * 1)

        return macs

    def params(self):
        params = 0
        params += 1 * 1 * self.channel * self.channel//self.reduction
        params += 1 * 1 * self.channel//self.reduction * self.planes
        return params
 


class SRGFS_Inf(nn.Module):
    def __init__(self, f, scale, n_residual, s):
        super(SRGFS_Inf, self).__init__()
        self.s = s
        self.f = f
        self.scale = scale
        self.feature_extra = nn.Sequential(
            nn.Conv2d(1, f, 3, 1, 1),
            nn.PReLU(num_parameters=f)
            )
        self.fnl_conv = nn.Sequential(
            nn.Conv2d(f, scale*scale, 3, 1, 1)
        )
        self.gffs_blocks = self.make_layers(GFFS_Inf, n_residual, f, self.s)
        self.upSampling = nn.PixelShuffle(scale)

        self._init_weights()

    def freeze_layers(self):
        for name, m in self.named_children():
            if not name.endswith('out'):
                for params in m.parameters():
                    params.requires_grad = False

    def _init_weights(self):
        for m in self.gffs_blocks:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
        for m in self.feature_extra:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
        for m in self.fnl_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
        

    def make_layers(self, block, num_of_layers, f, s):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(f, s))
        return nn.Sequential(*layers)


    def forward(self, input):
        x, meta = input
        self.h, self.w = x.shape[2], x.shape[3]
        identity = x
        x_first = self.feature_extra(x)
        x, meta = self.gffs_blocks((x_first, meta))
        x = torch.add(x, x_first)
        x = self.fnl_conv(x)
        x = torch.add(x, identity)
        x = self.upSampling(x)
        return x, meta

    def macs(self):
        macs = 0
        # linearBlock_55_in
        macs = 9 * 1 * self.f * (self.h * self.w)
        # residual_layers
        for layer in self.gffs_blocks:
            macs += layer.macs()

        # linearBlock_55_out
        macs += 9 * self.f * (self.scale*self.scale) * (self.h * self.w)

        return macs

    def params(self):
        params = 0
        params += 3 * 3 * 1 * self.f
        for layer in self.gffs_blocks:
            params += layer.params()
        
        params += 3 * 3 * self.f * (self.scale*self.scale)

        return params

        
if __name__ == "__main__":
    meta = {'masks': [], 'general_mask':[], 'features':[], 'layer_features':[], 'tail_masks':[]}
    upscale = 4
    height = (1280 // upscale)
    width = (720 // upscale) 
    model = SRGFS_Inf(16, upscale, 5, 0.5)
    model.eval()
    x = torch.randn((1, 1, height, width))
    prior = x.clone()
    x,_ = model((x, meta))
    print(height, width, float(model.macs()) / 1e9)
    print(height, width, float(model.params()) / 1e3)
    print(x.shape)