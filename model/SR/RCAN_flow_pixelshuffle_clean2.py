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
from model.SR.common import ZeroPad
from model.SR.spynet_arch import SpyNet, flow_warp
from model.SR.common import Upsampler, default_conv
from loss.loss import *
from utils.utils import filter2D, simulation

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()

        conv = default_conv


        self.factor = args.scale_factor
        self.in_channels = args.in_channels
        n_resgroups = 8
        n_feats = 64
        kernel_size = 3
        n_resblocks = 20

        modules_clean = [
            conv(self.in_channels, n_feats, kernel_size),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualGroup(n_feat=n_feats, n_resblocks=n_resblocks),
            conv(n_feats, self.in_channels, kernel_size),
        ]
        self.clean = nn.Sequential(*modules_clean)
        is_fix_cleaning = 0
        if is_fix_cleaning:  # keep the weights of the cleaning module fixed
            self.clean.requires_grad_(False)

        self.network = RCAN(in_channels=args.in_channels, out_channels=1, scale_factor=4)
        # self.projection = Projection(args.scale_factor)

    def forward(self, input_data, info=None):
        

        # rays = input_data['rays']
        clean = input_data['input']


        for i in range(1):
            clean = self.clean(clean)

        # PSF = input_data['psf']
        # for i in range(3):
        #     out = self.network(rgbs)
        #     rgbs = self.projection(out, PSF)
        flow_result, out= self.network(clean)
        output = {}
        output['sr'] = out
        output['clean'] = clean
        output['flow_result'] = flow_result
        return output

## Channel Attention (CA) Layer
class RCAN(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(RCAN, self).__init__()

        self.spynet = SpyNet()

        conv = default_conv

        self.factor = scale_factor
        n_resgroups = 2
        n_feats = 32
        kernel_size = 3
        n_resblocks = 1
        self.n_feats = n_feats

        modules_head = [conv(in_channels, n_feats, kernel_size)]

        modules_body = [
            ResidualGroup(n_feat=n_feats, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_upscale = Upsampler(conv, self.factor, n_feats, act=False, type='pixelshuffle')



        # define tail module
        modules_last = [
            conv(n_feats, 1, kernel_size),
            nn.LeakyReLU(0.1, inplace=True),
        ]


        # define tail module
        modules_tail = [
            nn.LeakyReLU(0.1, inplace=True),
            conv(in_channels, n_feats, kernel_size),
            nn.LeakyReLU(0.1, inplace=True),
            conv(n_feats, 1, kernel_size)
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.upscale = nn.Sequential(*modules_upscale)

        self.conv_last = nn.Sequential(*modules_last)

        self.tail = nn.Sequential(*modules_tail)


        # self.img_upsample = nn.Upsample(
        #     scale_factor=4, mode='bilinear', align_corners=False)

        # upsample
        # self.fusion = nn.Conv2d(
        #     n_feats * 2, n_feats, 1, 1, 0, bias=True)


        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def get_flow(self, x):
        b, n, h, w = x.size()

        x_1 = x.reshape(-1, 1, h, w)
        x_2 = x[:,n//2:n//2+1,:,:].repeat(1,n,1,1).reshape(-1, 1, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n, 2, h, w)

        return flows_forward


    def forward(self, x):

        b, n, h, w = x.size()
        flows_backward = self.get_flow(x)
        feat_prop = x.new_zeros(b, self.n_feats, h, w)

        flow_result = []

        for i in range(0, n, 1):
            x_i = x[:, i:i+1,  :, :]           
            flow = flows_backward[:, i,  :, :]
            feat_prop = flow_warp(x_i, flow.permute(0, 2, 3, 1))
            flow_result.append(feat_prop)

        flow_result = torch.cat(flow_result, dim=1)

        x = self.head(flow_result)
        x = self.body(x) + x
        up_features = self.upscale(x) 
        result = self.tail(up_features)

        return flow_result, result



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
        self.tvloss = TVLoss(TVLoss_weight= 1)
        self.projection = Projection(args.scale_factor)
        self.FFTLoss = FFTLoss()

    def forward(self, input_data, output_data, criterion_data=[]):
    
        # LR = input_data['input']
        PSF = input_data['psf']
        GT = input_data['gt']
        mixed_kernel = input_data['mixed_kernel']
        SR = output_data['sr']
        input_clean = input_data['input_clean']
        # up_features = output_data['up_features']
        clean = output_data['clean']
        sinc_SR = filter2D(SR, mixed_kernel)

        # outputs = self.projection(sinc_SR, PSF) ## fft
        outputs = simulation(sinc_SR, PSF)
        outputs = outputs[:,:,::4,::4]

        loss = 0

        # denoise
        loss += self.criterion_Loss(input_clean, clean)
        loss += self.tvloss(SR) 

        # supervised 
        loss += self.FFTLoss(GT, SR) 
        loss += self.criterion_Loss(GT, SR)

        # unsupervised
        loss +=  self.FFTLoss(outputs, input_clean) 
        loss += self.criterion_Loss(outputs, input_clean)

        # loss +=  self.FFTLoss(input_data['hr_stack'], output_data['sr_stack']) 
        # loss += self.criterion_Loss(input_data['hr_stack'], output_data['sr_stack'])
        b, n, h, w = input_clean.size()
        flow_result = output_data['flow_result']
        flow_gt = input_clean[:,n//2:n//2+1,:,:].repeat(1,n,1,1)
        loss +=  self.FFTLoss(flow_result, flow_gt) 
        loss += self.criterion_Loss(flow_result, flow_gt)


        return loss


def weights_init(m):
    pass

