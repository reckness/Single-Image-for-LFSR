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
from loss.loss import *
from utils.utils import filter2D, simulation


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()

        conv = default_conv

        self.factor = args.scale_factor
        self.in_channels = args.in_channels
        n_resgroups = 1
        n_feats = 64
        kernel_size = 3
        n_resblocks = 10


        modules_clean = [
            conv(self.in_channels, n_feats, kernel_size),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualGroup(n_feat=n_feats, n_resblocks=n_resblocks),
            conv(n_feats, self.in_channels, kernel_size),
        ]

        modules_head = [conv(self.in_channels, n_feats, kernel_size)]


        modules_body = [
            ResidualGroup(n_feat=n_feats, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_upscale = Upsampler(conv, 2, n_feats, bn=False, act=False, type='pixelshuffle')

        # define tail module
        modules_tail = [
            conv(n_feats, 1, kernel_size)
        ]
        
        self.clean = nn.Sequential(*modules_clean)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.upscale = nn.Sequential(*modules_upscale)
        self.tail = nn.Sequential(*modules_tail)


        is_fix_cleaning = 0
        if is_fix_cleaning:  # keep the weights of the cleaning module fixed
            self.clean.requires_grad_(False)

        # self.upscale.requires_grad_(False)
            
    def forward(self, input_data, info=None):
        # rays = input_data['rays']
        clean = input_data['input']
        for i in range(1):
            clean = self.clean(clean)
        x = self.head(clean)
        x = self.body(x) + x
        x = self.upscale(x)  +  F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = self.body(x) + x
        x = self.upscale(x) +  F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = self.body(x) + x
        x = self.upscale(x)  +  F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        out = self.tail(x)
        output = {}
        output['sr'] = out
        output['clean'] = clean
        return output


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
        modules_body.append(nn.Conv2d(n_feat, 1, 1, 1, 0))
        modules_body.append(nn.LeakyReLU(0.1, inplace=True))        
        modules_body.append(nn.Conv2d(1, n_feat, 1, 1, 0))
        modules_body.append(nn.LeakyReLU(0.1, inplace=True))
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
        self.tvloss = TVLoss(TVLoss_weight= 1)
        self.projection = Projection(args.scale_factor)
        self.FFTLoss = FFTLoss()
        self.scale_factor = args.scale_factor

    def forward(self, input_data, output_data, criterion_data=[]):
    
        # LR = input_data['input']
        PSF = input_data['psf']
        SR = output_data['sr']
        input_clean = input_data['input_clean']
        # up_features = output_data['up_features']
        clean = output_data['clean']

        if 'mixed_kernel' in input_data.keys():
            mixed_kernel = input_data['mixed_kernel']
            blur_SR = filter2D(SR, mixed_kernel)

        # outputs = self.projection(sinc_SR, PSF) ## fft
        outputs = simulation(blur_SR, PSF)
        outputs = outputs[:,:,::self.scale_factor,::self.scale_factor]

        loss = 0

        # denoise
        loss += self.criterion_Loss(input_clean, clean)
        loss += self.tvloss(SR) 

        # supervised 
        if 'gt' in input_data.keys():
            GT = input_data['gt']
            loss += self.FFTLoss(GT, SR) 
            loss += self.criterion_Loss(GT, SR)

        # unsupervised
        loss +=  self.FFTLoss(outputs, input_clean) 
        loss += self.criterion_Loss(outputs, input_clean)

        return loss


def weights_init(m):
    pass

