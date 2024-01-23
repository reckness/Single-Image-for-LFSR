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

        modules_head = [conv(self.in_channels, n_feats, kernel_size)]

        modules_body = [
            ResidualGroup(n_feat=n_feats, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_upscale = Upsampler(conv, self.factor, n_feats, bn=False, act=False, type='pixelshuffle')

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

        # for i in range(3):
        #     residues = self.clean(clean)
        #     clean = clean + residues
        #     # determine whether to continue cleaning
        #     if torch.mean(torch.abs(residues)) < 0.1:
        #         break    

        for i in range(1):
            clean = self.clean(clean)
            # clean = clean + residues
            # # determine whether to continue cleaning
            # if torch.mean(torch.abs(residues)) < 0.1:
            #     break   


        x = self.head(clean)
        x = self.body(x) + x
        up_features = self.upscale(x) # +  F.interpolate(x, scale_factor=self.factor, mode='bicubic', align_corners=False)
        out = self.tail(up_features)
        output = {}
        output['sr'] = out
        output['up_features'] = up_features
        output['clean'] = clean
        return output


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RepVGGBlock(n_feat, n_feat, 3, padding=1) \
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
        self.scale_factor = args.scale_factor

    def forward(self, input_data, output_data, criterion_data=[]):
    
        # LR = input_data['input']
        PSF = input_data['psf']
        
        
        SR = output_data['sr']
        input_clean = input_data['input_clean']
        # input_resize = input_data['input_resize']
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



def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()


        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

