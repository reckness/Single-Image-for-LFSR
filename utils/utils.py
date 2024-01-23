import numpy as np
import os
from skimage import metrics
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from option import args
from einops import rearrange
import xlwt
import torch.nn.functional as F
from scipy import special
from PIL import Image
from skimage import io

from utils.imresize import *

class ExcelFile():
    def __init__(self):
        self.xlsx_file = xlwt.Workbook()
        self.worksheet = self.xlsx_file.add_sheet(r'sheet1', cell_overwrite_ok=True)
        self.worksheet.write(0, 0, 'Datasets')
        self.worksheet.write(0, 1, 'Scenes')
        self.worksheet.write(0, 2, 'PSNR')
        self.worksheet.write(0, 3, 'SSIM')
        self.worksheet.col(0).width = 256 * 16
        self.worksheet.col(1).width = 256 * 22
        self.worksheet.col(2).width = 256 * 10
        self.worksheet.col(3).width = 256 * 10
        self.sum = 1

    def write_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        for i in range(len(psnr_iter_test)):
            self.add_sheet(test_name, LF_name[i], psnr_iter_test[i], ssim_iter_test[i])

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        self.add_sheet(test_name, 'average', psnr_epoch_test, ssim_epoch_test)
        self.sum = self.sum + 1

    def add_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.6f' % psnr_iter_test)
        self.worksheet.write(self.sum, 3, '%.6f' % ssim_iter_test)
        self.sum = self.sum + 1


def get_logger(log_dir, args):
    '''LOG '''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def create_dir(args):
    log_dir = Path(args.path_log)
    log_dir.mkdir(exist_ok=True)
    if args.task == 'SR':
        task_path = 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.scale_factor) + 'x_' +str(args.mark)
    
    log_dir = log_dir.joinpath(task_path)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.data_name)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.model_name)
    log_dir.mkdir(exist_ok=True)

    checkpoints_dir = log_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    results_dir = log_dir.joinpath('results/')
    results_dir.mkdir(exist_ok=True)

    
    return log_dir, checkpoints_dir, results_dir


def create_dir_single(args):
    log_dir = Path(args.path_log)
    log_dir.mkdir(exist_ok=True)
    if args.task == 'SR':
        task_path = 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.scale_factor) + 'x_' +str(args.mark)
    
    log_dir = log_dir.joinpath(task_path)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.data_name)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.model_name)
    log_dir.mkdir(exist_ok=True)

    checkpoints_dir = log_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    results_dir = log_dir.joinpath('results/')
    results_dir.mkdir(exist_ok=True)

    
    return log_dir, checkpoints_dir, results_dir


class Logger():
    def __init__(self, log_dir, args):
        self.logger = get_logger(log_dir, args)

    def log_string(self, str):
        if args.local_rank <= 0:
            self.logger.info(str)
            print(str)


def cal_metrics(args, label, out,):

    B, C, h, w = label.size()
    PSNR = np.zeros(shape=(B, 1, 1), dtype='float32')
    SSIM = np.zeros(shape=(B, 1, 1), dtype='float32')
    label_y = label[:, 0, :, :].data.cpu()
    out_y = out[:, 0, :, :].data.cpu()
    for b in range(B):
        PSNR[b] = metrics.peak_signal_noise_ratio(label_y[b, :, :].numpy(), out_y[b, :, :].numpy(),data_range =1.0)

        SSIM[b] = metrics.structural_similarity(label_y[b, :, :].numpy(),
                                                        out_y[b, :, :].numpy(),
                                                        gaussian_weights=True, data_range=1.0)

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out


def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)

    return subLF


def Stackdivide(data, angRes, patch_size, stride):
    data = data.permute(1, 0, 2, 3)
    [C, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, 'view (h w) (n1 n2) -> (n1 n2) view h w',
                       h=patch_size, w=patch_size, n1=numU, n2=numV)
    return subLF, numU, numV


def LFintegrate_stack(subLF, pz, stride, h, w, numU, numV):
    bdr = (pz - stride) // 2
    outLF = subLF[:, :,  bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, '(n1 n2) 1  h w -> 1 1 (n1 h) (n2 w)', n1=numU, n2=numV)
    outLF = outLF[:, :, 0:h, 0:w]
    return outLF


def LFintegrate(subLF, angRes, pz, stride, h, w):
    # if subLF.dim() == 4:
    #     subLF = rearrange(subLF, 'n1 n2 h w -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
    #     pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :,  bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2  h w -> 1 1 (n1 h) (n2 w)')
    outLF = outLF[:, :, 0:h, 0:w]

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y


def loadmat(file):        
    try:
        import h5py
        data = h5py.File(file, 'r')
    except:
        import scipy.io as scio
        data = scio.loadmat(file)
    return data


def tiff3Dread(path):
    from PIL import Image
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        # tmp = np.stack((img,) * 3, axis=-1)
        images.append(np.array(img))
    return np.array(images)


def ZeroPad(img1, img2):
    """
        padding the  img2 to the same with img1
    """
    import torch.nn as nn
    _, _, w, h = img1.shape
    _, _, w1, h1 = img2.shape

    if img2.shape[2] %2 == 0:
        m = nn.ZeroPad2d((
            int((h-h1)/2), 
            int((h-h1)/2), 
            int((w-w1)/2), 
            int((w-w1)/2)))
    else:
        m = nn.ZeroPad2d((
            int((h-h1)/2), 
            int((h-h1)/2+1), 
            int((w-w1)/2), 
            int((w-w1)/2+1)))
    return m(img2)


def ZeroPad_np(img1, img2):
    """
        padding the  img2 to the same with img1
    """
    (w, h) = img1.shape
    (_, w1, h1) = img2.shape

    if (w-w1)  % 2 == 0:
        return np.pad(img2, ((0, 0),
            (int((h-h1)/2), 
            int((h-h1)/2)), 
            (int((w-w1)/2), 
            int((w-w1)/2))), 'constant')
    else:
        return np.pad(img2,((0, 0),
            (int((h-h1)/2), 
            int((h-h1)/2+1)), 
            (int((w-w1)/2), 
            int((w-w1)/2+1))), 'constant')


def rgb2gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.114]).astype(np.single)


def psf2otf(psf):
    """
        convert the single psf to otf
    """
    return np.fft.fft2(np.fft.ifftshift(psf))


def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D sinc filter
    Reference: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd.
        pad_to (int): pad kernel size to desired size, must be odd or zero.
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D
    add kernel rotate 180
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """

    # rotate kernel 180
    # kernel = torch.rot90(kernel, k=2, dims=[1,2])

    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        # kernel = kernel.view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def simulation(img, kernel, stride=1):
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, 1, h, w)
        kernel (Tensor): (b, c, k, k)
    """

    ### rotate the psf 180
    kernel = torch.rot90(kernel, k=2, dims=[2,3])

    _, c, k, _ = kernel.size()
    b, _, h, w = img.size()
    if k % 2 == 1:
        pad_size = ceil((k-stride) / 2)
        img = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]
    img = img.repeat(1, c, 1, 1)        
    img = img.view(1, b * c, ph, pw)
    kernel = kernel.view(b * c, 1, k, k)
    
    return F.conv2d(img, kernel, groups=b * c, stride=(stride, stride)).view(b, c, h//stride, w//stride)




def random_add_gaussian_noise_pt(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    # https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/data/degradations.py#L544
    noise = random_generate_gaussian_noise_pt(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out

def random_generate_gaussian_noise_pt(img, sigma_range=(0, 10), gray_prob=0):
    sigma = torch.rand(
        img.size(0), dtype=img.dtype, device=img.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_gaussian_noise_pt(img, sigma, gray_noise)


def generate_gaussian_noise_pt(img, sigma=10, gray_noise=0):
    """Add Gaussian noise (PyTorch version).
    Args:
        img (Tensor): Shape (b, c, h, w), range[0, 1], float32.
        scale (float | Tensor): Noise scale. Default: 1.0.
    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = img.size()
    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(img.size(0), 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        noise_gray = torch.randn(*img.size()[2:4], dtype=img.dtype, device=img.device) * sigma #/ 255.
        noise_gray = noise_gray.view(b, 1, h, w)

    # always calculate color noise
    noise = torch.randn(*img.size(), dtype=img.dtype, device=img.device) * sigma #/ 255.

    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise





def MacPI2SAI(x, angRes):
    b, c, hu, wv = x.shape
    out = []
    for i in range(angRes):
        # out_h = []
        for j in range(angRes):
            # out_h.append(x[:, :, i::angRes, j::angRes])
            out.append(x[:, :, i:hu+angRes+1:angRes, j:wv+angRes+1:angRes])
        # out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 1)
    return out


# def SAI2MacPI(x, angRes):
#     b, c, hu, wv = x.shape
#     h, w = hu // angRes, wv // angRes
#     tempU = []
#     for i in range(h):
#         tempV = []
#         for j in range(w):
#             # tempV.append(x[:, :, i::h, j::w])
#             tempV.append(x[:, :, i:hu + 5:h, j:wv + 5:w])
#         tempU.append(torch.cat(tempV, dim=3))
#     out = torch.cat(tempU, dim=2)
#     return out


def load_psf(psf_path, angRes, down_sample=1):
    """load light field psf.
    Args:
        psf_path: 
        angRes: angular resolution
    Returns:
        psf_stack: Shape (angRes*angRes, w, h)
    """

    psf_stack = loadmat(psf_path)
    if 'psf_z' in psf_stack.keys():
        psf_stack = psf_stack['psf_z']
    elif 'PSF_stack' in  psf_stack.keys():
        psf_stack = psf_stack['PSF_stack']
    else:
        psf_stack = psf_stack['psf']['psf_z'][0][0]
    psf_stack = np.array(psf_stack)

    tmp = []
    if psf_stack.shape[0] == angRes:
        for u in range(angRes):
            for v in range(angRes):
                # tmp.append(psf_stack[u, v, :, :])
                psf = imresize(psf_stack[u, v,:,:], scalar_scale=down_sample)
                tmp.append(psf)
    elif psf_stack.shape[2] == angRes:
        for u in range(angRes):
            for v in range(angRes):
                psf = imresize(psf_stack[:, :, u, v], scalar_scale=down_sample)
                tmp.append(psf)
    else:
        return 0 

    psf_stack = np.stack(tmp)

    rows, cols = psf_stack.shape[-2:]

    # crop the psf to save the computation
    psf_stack = psf_stack/np.max(psf_stack)
    index = np.where(psf_stack > 5e-4)
    index_min_x = np.min(index[1])
    index_max_x = np.max(index[1])
    index_min_y = np.min(index[2])
    index_max_y = np.max(index[2])  

    border_size = min(index_min_x, cols-index_max_x, index_min_y, rows-index_max_y)

    psf_size = rows-2*border_size

    if psf_size % 2 == 0:
        psf_size = psf_size - 1
    psf_stack = psf_stack[:, border_size:border_size+psf_size, border_size:border_size+psf_size]

    # psf enengy normalization  each view
    tmp = np.sum(psf_stack[angRes*(angRes//2-1)+angRes//2-1,:,:])
    for u in range(angRes*angRes):
        psf_stack[u,:,:] = psf_stack[u,:,:]/np.sum(psf_stack[u,:,:]/tmp)

    return psf_stack


def conbines(input_path,output_path,angRes):
    # get file name
    image_path = os.listdir(input_path)
    for path in image_path: 
        save_name = ''
        save_name = (path).split('.')[-1]
        # print(save_name)
        if  not os.path.exists(output_path):
            os.makedirs(output_path)
        save_path = output_path + save_name +'.tif'
        print(save_path)
        image_files = [f for f in os.listdir(os.path.join(input_path,path)) if os.path.isfile(os.path.join(input_path, path, f))]
        # load the image width and height
        frist_image_path = os.path.join(input_path,path ,image_files[0])
        frist_image = Image.open(frist_image_path)
        width,height = frist_image.size
        #creat NumPY
        combined_image = np.zeros((angRes*angRes,height,width),dtype= np.uint8)
        if len(image_files) != angRes * angRes:
           print('the number of image wrong!')
        else:
           for i ,image_file in enumerate(image_files):
            # print(image_file)
            image_path = os.path.join(input_path,path ,image_file)
            # print(image_path)
            img = Image.open(image_path)
            img_data = np.array(img)
            combined_image[i ,: ,:] = img_data
        io.imsave(save_path,combined_image,plugin='tifffile')


def pre_img (x,angRes):
    c,h,w = x.shape
    # print('x.shape:',x.shape)
    output_img = np.zeros((1,h*angRes,w*angRes),dtype ='single')
    # x = x[: , 5:6,: ,:]
    # print('x',x.shape)
    # print('out',output_img.shape)
    count = 0
    for u in range(angRes):
        for v in range(angRes):
            # print('xxxx',x.shape)
            output_img[:,u*h:(u+1)*h,v*w:(v+1)*w] = x[count:count+1,:,:]
            count += 1
            # print(x.shape)
    return output_img

def pre_img_Train (x,angRes):
    b,c,h,w = x.shape
    # print(x.shape)
    output_img = torch.empty(1,1,h*angRes,w*angRes)
    # x = x[: , 5:6,: ,:]
    # print('x',x.shape)
    # print('out',output_img.shape)
    count = 0
    for u in range(angRes):
        for v in range(angRes):
            # x = x.unsqueeze(1)
            output_img[:,:,u*h:(u+1)*h,v*w:(v+1)*w] = x[ :,count:count+1,:,:]
            count += 1
            # print(x.shape)
    return output_img
        