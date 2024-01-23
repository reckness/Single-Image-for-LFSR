import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import torch.nn.functional as F
from loss.loss import *
from utils.utils import filter2D, simulation



class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.input_nc = args.in_channels
        self.output_nc = 1
        self.ngf = args.channels
        norm_layer=nn.BatchNorm2d
        use_dropout=False
        n_blocks=args.n_block
        padding_type='reflect'
        self.angRes = args.angRes_in
        self.factor = args.scale_factor

        use_bias = True

        self.cleaning = Inconv(self.input_nc, self.input_nc, norm_layer, use_bias)


        self.inc = Inconv(self.input_nc, self.ngf, norm_layer, use_bias)
        self.down1 = Down(self.ngf, self.ngf * 2, norm_layer, use_bias)
        self.down2 = Down(self.ngf * 2, self.ngf * 4, norm_layer, use_bias)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(self.ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)

        self.up1 = Up(self.ngf * 4, self.ngf * 2, norm_layer, use_bias)
        self.up2 = Up(self.ngf * 2, self.ngf, norm_layer, use_bias)

        upsample = [
            nn.Conv2d(self.ngf, self.ngf*self.factor**2, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True),            
            nn.PixelShuffle(self.factor),
            nn.Conv2d(self.ngf, self.ngf, 3, stride=1, padding=1, bias=True),
        ]

        self.upsample = nn.Sequential(*upsample)
        self.outc = Outconv(self.ngf, self.output_nc)

    def forward(self, input_data, info=None):

        x = input_data['input']
        clean = self.cleaning(x)

        out = self.inc(clean)
        res = out 
        out = self.down1(out)
        out = self.down2(out)
        out = self.resblocks(out)
        out = self.up1(out)
        out = self.up2(out) + res
        out = self.upsample(out) 
        sr = self.outc(out)

        output = {}
        output['sr'] = sr
        output['clean'] = clean        

        return output



class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
                      bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=2, padding=1, bias=use_bias),
            # norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)
                       ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(in_ch, out_ch,
            #           kernel_size=3, stride=1,
            #           padding=1, bias=use_bias),
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=use_bias),
            # norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x
    

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
        input_clean = input_data['input_clean']

        SR = output_data['sr']
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

        return loss


def weights_init(m):
    pass