import torch
import torch.nn as nn
from model.SR.common import ZeroPad
import torchvision

class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)            
        return loss 


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class Projection(nn.Module):
    def __init__(self, factor):
        super(Projection, self).__init__()
        self.factor = factor

    def forward(self, sr, psf):
        PadPSF = ZeroPad(sr, psf)
        sr_fft = torch.fft.fft2(torch.fft.fftshift(sr))
        otf = torch.fft.fft2(torch.fft.fftshift(PadPSF))
        sr_fft_t = torch.mul(sr_fft, otf)
        outputs = torch.fft.fftshift(torch.fft.ifft2(sr_fft_t))
        outputs = outputs.real
        outputs= outputs[:,:,::self.factor,::self.factor]
        return outputs

class ConvProjection(nn.Module):
    def __init__(self, factor):
        super(Projection, self).__init__()
        self.factor = factor

    def forward(self, sr, psf):
        PadPSF = ZeroPad(sr, psf)
        sr_fft = torch.fft.fft2(torch.fft.fftshift(sr))
        otf = torch.fft.fft2(torch.fft.fftshift(PadPSF))
        sr_fft_t = torch.mul(sr_fft, otf)
        outputs = torch.fft.fftshift(torch.fft.ifft2(sr_fft_t))
        outputs = outputs.real
        outputs= outputs[:,:,::self.factor,::self.factor]
        return outputs


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
 
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class FFTLoss(torch.nn.Module):
    """FFT Loss"""
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.criterion_Loss = nn.L1Loss()
 
    def forward(self, source, target):
        return self.criterion_Loss(torch.fft.fft2(source), torch.fft.fft2(target))


class WaveletLoss(torch.nn.Module):
    """WaveletLoss"""
    def __init__(self):
        super(WaveletLoss, self).__init__()
        self.criterion_Loss = nn.L1Loss()
        from pytorch_wavelets import DWTForward, DWTInverse
        self.xfm = DWTForward(J=8, mode='zero', wave='haar').cuda()

 
    def forward(self, source, target):


        Yl_s, Yh_s = self.xfm(source)
        Yl_t, Yh_t = self.xfm(target)

        loss = 0
        loss += self.criterion_Loss(Yl_s, Yl_t)
        loss += self.criterion_Loss(Yh_s[0], Yh_t[0])

        return loss

