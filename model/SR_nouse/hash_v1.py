'''
@Article{DistgLF,
    author    = {Wang, Yingqian and Wang, Longguang and Wu, Gaochang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
    title     = {Disentangling Light Fields for Super-Resolution and Disparity Estimation},
    journal   = {IEEE TPAMI},
    year      = {2022},
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SR import common
import cv2
# import commentjson as json
# import tinycudann as tcnn
import numpy as np
import os
from model.SR.common import ZeroPad
from encoding.positional_encoding import get_embedder
from loss.loss import *


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = args.channels
        n_group = args.n_group
        n_block = args.n_block
        kernel_size = 3
        self.angRes = args.angRes_in
        self.factor = args.scale_factor
        # self.network =  tcnn.NetworkWithInputEncoding(
        #         n_input_dims=2, n_output_dims=1,
        #         encoding_config={
        #             "otype": "Grid",           #// Component type.
        #             "type": "Hash",            #// Type of backing storage of the grids. Can be "Hash", "Tiled" or "Dense".
        #             "n_levels": 16,           # // Number of levels (resolutions)
        #             "n_features_per_level": 2, #// Dimensionality of feature vector stored in each level's entries.
        #             "log2_hashmap_size": 19,   #// If type is "Hash", is the base-2 logarithm of the number of elements in each backing hash table.
        #             "base_resolution": 16,    # // The resolution of the coarsest level is base_resolution^input_dims.
        #             "per_level_scale": 1.5,    #// The geometric growth factor, i.e.the factor by which the resolution of each grid is larger (per axis) than that of the preceeding level.
        #             "interpolation": "Linear"}, #// How to interpolate nearby grid lookups. Can be "Nearest", "Linear", or "Smoothstep" (for smooth deri-vatives).,                
        #         network_config={
        #             "otype": "CutlassMLP",
        #             "activation": "ReLU",
        #             "output_activation": "None",
        #             "n_neurons": 128,
        #             "n_hidden_layers": 2,
        #         }
        #     )
        self.network, _ = get_embedder(10, 0)

        self.mlp1 = torch.nn.Linear(42, # 输入的神经元个数
           256, # 输出神经元个数
           bias=True # 是否包含偏置
        )
        self.activate = nn.Tanh()


        self.mlp2 = torch.nn.Linear(256+42, # 输入的神经元个数
           1, # 输出神经元个数
           bias=True # 是否包含偏置
        )


    def forward(self, input_data, info=None):
        

        rays = input_data['rays']
        # rgbs = input_data['masked_input']
        
        (b, w, h, c) = rays.shape
        x = rays.reshape(b*w*h,c)
        x = self.network(x)
        # import ipdb
        # ipdb.set_trace()
        res = self.mlp1(x)
        x = torch.cat([res, x], dim=-1)
        x = self.activate(x)
        x = self.mlp2(x)
        x = x.reshape(b,-1,w,h).float()

        # import ipdb
        # ipdb.set_trace()

        output = {}
        output['sr'] = x


        
        return output




class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()
        self.tvloss = TVLoss(TVLoss_weight= 0.5)
        self.projection = Projection(args.scale_factor)

        self.FFTLoss = FFTLoss()


    def forward(self, input_data, output_data, criterion_data=[]):

        LR = input_data['input']
        PSF = input_data['psf']
        SR = output_data['sr']
        outputs = self.projection(SR, PSF) ## fft


        loss =  self.FFTLoss(LR, outputs) #+ self.tvloss(SR)

        return loss

def weights_init(m):
    pass


