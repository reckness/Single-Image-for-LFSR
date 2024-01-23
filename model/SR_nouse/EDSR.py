'''
@inproceedings{EDSR,
  title={Enhanced deep residual networks for single image super-resolution},
  author={Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Mu Lee, Kyoung},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={136--144},
  year={2017}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()

        channel = 64
        self.factor = args.scale_factor
        self.init_feature = nn.Conv2d(32, channel, 3, 1, 1)
        self.body = ResidualGroup(channel, 8)
        if args.scale_factor == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(channel, channel * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(channel, 1, 3, 1, 1))
        if args.scale_factor == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(channel, channel * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(channel, channel * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(channel, channel, 3, 1, 1))
        self.body2 = nn.Conv2d(channel, 32, 3, 1, 1)

        self.tail = nn.Conv2d(32, 1, 3, 1, 1)


    def forward(self, x, psf, info=None):
        # refocus = torch.sum(x, dim=1, keepdim=True)/32
        # # refocus = x[:,31:32,:,:]
        # refocus = torch.sum(x, dim=1, keepdim=True)/32
        res = F.interpolate(x, scale_factor=self.factor, mode='bicubic', align_corners=False)


        buffer = self.init_feature(x)
        # buffer = self.upscale(buffer)
        buffer = F.interpolate(buffer, scale_factor=self.factor, mode='bicubic', align_corners=False)

        buffer = self.body(buffer)

        buffer = self.body2(buffer) + res
        out = self.tail(buffer) #+ refocus

        return out


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            ResB(n_feat) \
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResB(nn.Module):
    def __init__(self, n_feat):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
        )


    def forward(self, x):
        res = 0.1 * self.body(x)
        res = res + x
        return res

def ZeroPad(img1, img2):
    """
        padding the  img2 to the same with img1
    """

    _, _, w, h = img1.shape
    _, _, w1, h1 = img2.shape
    m = nn.ZeroPad2d((int((w-w1)/2), int((w-w1)/2), int((h-h1)/2), int((h-h1)/2)))
    return m(img2)


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, LR, SR, HR, PSF, criterion_data=[]):


        if 1:
            B, C, W, H = PSF.shape

            PadPSF = ZeroPad(SR, PSF)
            PadPSF = PadPSF.cuda()

            outputs = []

            for b in range(B):
                channel_outputs = []
                sr_i = SR[b,0,:,:]            
                for c in range(C):
                    psf_i = PadPSF[b,c,:,:]
                    sr_i_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(sr_i)))
                    otf = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(psf_i)))
                    sr_i_fft_t = torch.mul(sr_i_fft, otf)
                    output_i = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(sr_i_fft_t)))

                    channel_outputs.append(output_i)
                channel_outputs = torch.stack(channel_outputs) 
                channel_outputs = channel_outputs.real
                outputs.append(channel_outputs)

            outputs = torch.stack(outputs) 

            outputs= outputs[:,:,0:outputs.shape[2]-1:4,0:outputs.shape[3]-1:4]
            # loss =  self.criterion_Loss(torch.fft.fft2(LR), torch.fft.fft2(outputs)) 
            loss =  self.criterion_Loss(LR, outputs)  
            return loss
        else:
            # import ipdb
            # ipdb.set_trace()
            loss = self.criterion_Loss(SR, HR)
            return loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    pass

