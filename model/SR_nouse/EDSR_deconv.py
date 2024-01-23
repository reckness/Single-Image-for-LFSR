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

from loss.loss import TVLoss, Projection

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()

        channel = 64

        self.init_feature = nn.Conv2d(4, channel, 3, 1, 1)
        self.body = ResidualGroup(channel, 32)
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
        self.tail = nn.Conv2d(channel, 1, 3, 1, 1)


    def forward(self, x, Lr_Info):
        buffer = self.init_feature(x)
        buffer = self.upscale(buffer)
        buffer = self.body(buffer)
        out = self.tail(buffer)

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


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()
        self.tvloss = TVLoss(TVLoss_weight= 0.2)
        self.projection = Projection(args.scale_factor)

    def forward(self, input_data, SR, criterion_data=[]):

        LR = input_data['input']
        PSF = input_data['psf']
        GT = input_data['gt']
        # outputs = self.projection(SR, PSF)
        # loss =  self.criterion_Loss(outputs, LR)+ self.tvloss(SR)

        loss = self.criterion_Loss(SR, GT)
        return loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    pass

