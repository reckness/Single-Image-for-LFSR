import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def conv_3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size,kernel_size,kernel_size), stride=1, padding=(kernel_size//2))

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    

def ZeroPad(img1, img2):
    """
        padding the  img2 to the same with img1
    """

    _, _, w, h = img1.shape
    _, _, w1, h1 = img2.shape

    if img2.shape[2] %2 == 0:
        # m = nn.ZeroPad2d((int((w-w1)/2), int((w-w1)/2), int((h-h1)/2), int((h-h1)/2)))
        m = nn.ZeroPad2d((
            int((h-h1)/2), 
            int((h-h1)/2), 
            int((w-w1)/2), 
            int((w-w1)/2)))
    else:
        # m = nn.ZeroPad2d((int((w-w1)/2), int((w-w1)/2+1), int((h-h1)/2), int((h-h1)/2+1)))
        m = nn.ZeroPad2d((
            int((h-h1)/2), 
            int((h-h1)/2+1), 
            int((w-w1)/2), 
            int((w-w1)/2+1)))
    return m(img2)


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out



class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True, type='pixelshuffle'):

        m = []
        if type == 'pixelshuffle':
            if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
                for _ in range(int(math.log(scale, 2))):
                    m.append(conv(n_feat, 4 * n_feat, 3, bias))
                    m.append(nn.PixelShuffle(2))
                    if bn: m.append(nn.BatchNorm2d(n_feat))
                    if act: m.append(act())
            elif scale == 3:
                m.append(conv(n_feat, 9 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(3))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
            else:
                raise NotImplementedError
        elif type == 'transpose':
            if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
                for _ in range(int(math.log(scale, 2))):
                    m.append(conv(n_feat, n_feat, 3, bias))
                    m.append(nn.ConvTranspose2d(64, 64,
                    kernel_size=4, stride=2,
                    padding=1, output_padding=0,
                    bias=True))
                    if bn: m.append(nn.BatchNorm2d(n_feat))
                    if act: m.append(act())
        elif type == 'fixed_transpose':
            if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
                for _ in range(int(math.log(scale, 2))):
                    m.append(conv(n_feat, n_feat, 3, bias))
                    m.append(RandomFixedConvTrans2d(in_channels=64, out_channels=64))
                    if bn: m.append(nn.BatchNorm2d(n_feat))
                    if act: m.append(act())
        elif type == 'pixelshuffle_ICNR':
            if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
                for _ in range(int(math.log(scale, 2))):
                    m.append(PixelShuffle_ICNR(n_feat, n_feat))
                    if bn: m.append(nn.BatchNorm2d(n_feat))
                    if act: m.append(act())
            # elif scale == 3:
            #     m.append(conv(n_feat, 9 * n_feat, 3, bias))
            #     m.append(nn.PixelShuffle(3))
            #     if bn: m.append(nn.BatchNorm2d(n_feat))
            #     if act: m.append(act())
            else:
                raise NotImplementedError
            
        super(Upsampler, self).__init__(*m)




class RandomFixedConvTrans2d(nn.Module):
    """
    A tranposed fixed random convolution.
    Wrapper for a more convenient use.
    Basically create a new convolution, initialize it and set
    `requires_grad_(False)` flag.
    """
    def __init__(self, in_channels=64, out_channels=64, kernel_size=4, stride=2,
                 padding=1, output_padding=0,
                 groups=1, dilation=1,
                 bias=True, padding_mode='zeros'):
        super(RandomFixedConvTrans2d, self).__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride,
            padding, output_padding,
            groups, bias, dilation, padding_mode)
        self.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)



class PixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    def __init__(self, ni:int, nf:int=None, scale:int=2, blur:bool=True, leaky:float=None, **kwargs):
        super().__init__()
        # nf = ifnone(nf, ni)
        self.conv = default_conv(ni, nf*(scale**2), 3)
        icnr_init(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
