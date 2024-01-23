'''
@inproceedings{RCAN,
  title={Image super-resolution using very deep residual channel attention networks},
  author={Zhang, Yulun and Li, Kunpeng and Li, Kai and Wang, Lichen and Zhong, Bineng and Fu, Yun},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={286--301},
  year={2018}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from model.SR.common import Upsampler, default_conv
from loss.loss import TVLoss, Projection


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()

        conv = default_conv

        self.factor = args.scale_factor
        n_resgroups = 6
        n_feats = 64
        kernel_size = 3
        n_resblocks = 10

        modules_head = [conv(32, n_feats, kernel_size)]

        modules_body = [
            ResidualGroup(n_feat=n_feats, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_upscale = Upsampler(conv, self.factor, n_feats, act=False, type='pixelshuffle')

        # define tail module
        modules_tail = [
            conv(n_feats, 1, kernel_size)
        ]
        
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.upscale = nn.Sequential(*modules_upscale)
        self.tail = nn.Sequential(*modules_tail)
            
    def forward(self, input_data, info=None):
        # rays = input_data['rays']
        rgbs = input_data['input']
        x = self.head(rgbs)
        x = self.body(x) + x
        x = self.upscale(x)
        x = self.tail(x)
        return x



## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, 4, 1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(4, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
            if i == 0: modules_body.append(nn.LeakyReLU(0.1, inplace=True))
        modules_body.append(CALayer(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(n_feat) \
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()
        self.tvloss = TVLoss(TVLoss_weight= 0.15)
        self.projection = Projection(args.scale_factor)

    def forward(self, input_data, SR, criterion_data=[]):

        # LR = input_data['input']
        # PSF = input_data['psf']
        GT = input_data['gt']
        # outputs = self.projection(SR, PSF)
        loss =  self.criterion_Loss(GT, SR)# + self.tvloss(SR)

        return loss

def weights_init(m):
    pass

