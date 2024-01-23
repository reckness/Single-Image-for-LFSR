import os
from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms

import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
import math
from torch.utils.data import DataLoader
from utils import *
from einops import rearrange
import scipy.io as scio
import imageio
from utils.utils import loadmat, tiff3Dread, ZeroPad_np, circular_lowpass_kernel, MacPI2SAI,pre_img_Train,pre_img
from utils.utils import random_add_gaussian_noise_pt, filter2D, simulation, load_psf
from utils.degradations import random_mixed_kernels
import torch.nn.functional as F
import torch.nn as nn
import cv2

class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.scale_factor = args.scale_factor
        self.dataset_dir = args.path_for_train 
        self.gaussian_noise_prob = 1   
        self.hr_patch_size = 32
        if self.scale_factor == 6:
            self.hr_patch_size = 276

        self.noise_std_max = args.noise_std_max
        self.stack_input = args.stack_input
        self.repeat = args.repeat
        self.weight = args.weight

        ''' load data filenames'''
        if args.data_name == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = args.data_name.split(',')
        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]                
            self.file_list.extend(tmp_list)
        
        if args.train_data_num == -1:
            self.item_num = len(self.file_list)
        else:
            self.file_list = self.file_list[:args.train_data_num]
            self.item_num = args.train_data_num 

        '''load light field psf'''
        self.psf_dir =  args.psf_path  #+str(args.angRes_in) + 'x' + str(args.angRes_in)+'/'
        if args.psf_name == 'ALL':
            self.psf_list = os.listdir(self.psf_dir)
        else:
            self.psf_list = args.psf_name.split(',')
        self.psf_nums = len(self.psf_list)
        print('nums of psf',self.psf_nums)
        self.psfs = []
        for path in self.psf_list:
            if args.psf_name == 'ALL':
                tmp_list = os.path.join(self.psf_dir, path)
            else:
                tmp_list = os.path.join(self.psf_dir, path + '.mat')
            psf_stack = load_psf(tmp_list, self.angRes_in, 1/args.psf_downsample)
            print(tmp_list, psf_stack.shape)
            self.psfs.append(psf_stack)

        ''' pulse kernel ''' 
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        ''' random mixed kernel config '''
        self.mixed_prob = args.mixed_prob
        self.kernel_range=[7, 9, 11, 13, 15, 17]
        self.kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
        ]
        self.kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1]
        self.sigma_x=[0.2, 3]
        self.sigma_y=[0.2, 3]
        self.beta_gaussian=[0.5, 4]

        self.transform = transforms.Compose([
            # transforms.CenterCrop((256,256))
            transforms.RandomCrop((self.hr_patch_size,self.hr_patch_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

    def __getitem__(self, index):

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        ''' load hr patch by index'''
        file_name = [self.dataset_dir + self.file_list[int(index / self.repeat)]]
        with h5py.File(file_name[0], 'r') as hf:
            Hr = np.array(hf.get('hr_center')) # Hr_SAI_y
            Hr = torch.FloatTensor(Hr.copy()).unsqueeze(0).unsqueeze(0)
            ''' crop high resolution image '''        
            Hr = self.transform(Hr)
            

        ''' random load the light field psf '''
        LFPSF = self.psfs[np.random.randint(self.psf_nums)]
        
        ''' filter the wigner by weight (eg., from 64 to 32 views) '''
        if self.stack_input:
            LFPSF = LFPSF[self.weight]

        ''' init random mixed kernel'''
        prob = np.random.uniform() * 100
        if prob < self.mixed_prob:
            kernel_size = random.choice(self.kernel_range)
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.sigma_x,
                self.sigma_y, [-math.pi, math.pi],
                self.beta_gaussian,
                self.beta_gaussian,
                noise_range=None)
            pad_size = (21 - kernel_size) // 2
            mixed_kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        else:
            mixed_kernel = self.pulse_tensor
        mixed_kernel = torch.FloatTensor(mixed_kernel)
        LFPSF = torch.FloatTensor(LFPSF.copy())

        ''' blur HR image by random mixed kernel '''
        hr_blur = filter2D(Hr, mixed_kernel)
        
        ''' simulation the low-resolution light field image by blur HR image '''
        input_clean = simulation(hr_blur, LFPSF.unsqueeze(0), stride=self.scale_factor)
        Hr = simulation(Hr,LFPSF.unsqueeze(0),stride = 1)
        Hr = pre_img_Train(Hr,Lr_angRes_in)
        # print('Hr:',Hr.shape)
        
        ''' add random gaussian noise '''
        input_noise = input_clean
        if np.random.uniform() < self.gaussian_noise_prob and self.noise_std_max>0:
            sigma =  np.random.randint(0, self.noise_std_max) / 1000
            input_noise = random_add_gaussian_noise_pt(input_clean, sigma_range=(sigma, sigma), gray_prob=0)
            input_noise = pre_img_Train(input_noise,Lr_angRes_in)
            # print('INPUT_NOise:',input_noise.shape)
            

        ''' pad the light field psf to same size'''
        kernel_size = LFPSF.shape[1]
        pad_size = (31 - kernel_size) // 2
        pad = (pad_size, pad_size, pad_size, pad_size)
        LFPSF = F.pad(LFPSF, pad, 'constant', 0)

        data = {
            'input': input_noise.squeeze(0),        # low resolution light field input with noise
            'input_clean': input_clean.squeeze(0),  # low resolution light field input without noise
            # 'hr_stack': hr_stack.squeeze(0),        # high resolution light field input without noise
            'gt': Hr.squeeze(0),                    # high resolution image without noise
            'psf': LFPSF,                           # light field psf
            'mixed_kernel': mixed_kernel
        }
        return data, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num * self.repeat


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    args.data_name = 'ALL'
    data_list = None
    if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
        if args.task == 'SR':
            dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.scale_factor) + 'x_08/'
            data_list = os.listdir(dataset_dir)
        elif args.task == 'RE':
            dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
            data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL', Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x_08/'
            self.data_list = [data_name]
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
            self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)
        self.stack_input = args.stack_input
        
        ''' weight '''
        self.weight = args.weight
        
        
    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y'))
            LFPSF = np.array(hf.get('psf_stack'))
            
        LFPSF = torch.from_numpy(LFPSF.copy())
        Lr_SAI_y = torch.from_numpy(Lr_SAI_y.copy()) 
        Hr_SAI_y = torch.from_numpy(Hr_SAI_y.copy()) 
        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        if self.stack_input:
            Lr_SAI_y = Lr_SAI_y[self.weight]
            LFPSF = LFPSF[self.weight]
        Lr_SAI_y_in = pre_img(Lr_SAI_y,Lr_angRes_in)
        Hr_SAI_y = pre_img(Hr_SAI_y,Lr_angRes_in)
        # print('LR:',Lr_SAI_y.shape)
        rays = get_rays(Hr_SAI_y.shape[1], Hr_SAI_y.shape[2])

        data = {
            'rays' : rays,
            'masked_input': Lr_SAI_y_in,
            'input': Lr_SAI_y_in,
            'psf': LFPSF,
            'gt':Hr_SAI_y,
            'restore':Lr_SAI_y
        }

        return data,  [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num 


class TestRealSingleDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL', Lr_Info=None):
        super(TestRealSingleDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.dataset_dir = args.path_for_single
        self.repeat = args.repeat
        self.item_num = 1
        self.stack_input = args.stack_input
        self.weight = args.weight
        self.psf_path = args.psf_path
        self.psf_name = args.psf_name

        '''' load light field point speard funciton'''
        self.psf_dir =  args.psf_path +str(args.angRes_in) + 'x' + str(args.angRes_in)+'/'


        PSF_PATH = os.path.join(self.psf_dir, self.psf_name + '.mat')
        LR_DATA_PATH = os.path.join(self.dataset_dir)

        psf_stack = load_psf(PSF_PATH, self.angRes_in)

        '''' load low-resolution light field image'''

        Num = self.angRes_in

        # read sub aperture images 
        if Num == 8:
            IMG_PATH = os.path.join(LR_DATA_PATH, 'view_0.tif')
        else:
            IMG_PATH = os.path.join(LR_DATA_PATH, 'view_001.tif')

        img = cv2.imread(IMG_PATH)

        OFFSET_X, OFFSET_Y = 0, 0
        PATCH_SIZE_H, PATCH_SIZE_W, _ = img.shape

        test_data = np.zeros([Num*Num, PATCH_SIZE_H, PATCH_SIZE_W])

        for u in range(0, Num):
            for v in range(0, Num):
                if Num == 8:
                    IMG_PATH = os.path.join(LR_DATA_PATH, 'view_'+ str(u*Num+v)+'.tif')
                else:
                    IMG_PATH = os.path.join(LR_DATA_PATH, 'view_'+ "{:0>3d}".format(u*Num+v+1)+'.tif')
                img = cv2.imread(IMG_PATH)
                if len(img.shape) == 3:
                    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.array(img).astype('float')
                test_data[u*Num+v, :,:] = img[OFFSET_X:OFFSET_X+PATCH_SIZE_H, OFFSET_Y:OFFSET_Y+PATCH_SIZE_W]

        self.lr = test_data
        self.psf_stack = psf_stack
        self.LF_name = LR_DATA_PATH.split('/')[-1]


    def __getitem__(self, index):

        Lr = torch.from_numpy(self.lr.copy()) 
        psf_stack = torch.from_numpy(self.psf_stack.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        rays = get_rays(Lr.shape[1]* 4, Lr.shape[2]* 4)

        if self.stack_input:
            Lr = Lr[self.weight]
            psf_stack = psf_stack[self.weight]

        rgbs = Lr.clone().float()

        data = {
            'rays' : rays,
            'masked_input': rgbs,
            'input': rgbs,
            'psf': psf_stack,
            # 'gt':Hr
        }


        return data, [Lr_angRes_in, Lr_angRes_out], self.LF_name
   
    def __len__(self):
        return self.item_num * self.repeat

class TestSingleDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL', Lr_Info=None):
        super(TestSingleDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.dataset_dir = args.path_for_single
        self.repeat = args.repeat
        self.item_num = 1
        self.stack_input = args.stack_input
        self.weight = args.weight


    def __getitem__(self, index):

        PSF_PATH = os.path.join(self.dataset_dir, 'kernelImage', 'PSF_stack.mat')
        LR_DATA_PATH = os.path.join(self.dataset_dir, 'blurredImage','000001.tif')
        HR_DATA_PATH = os.path.join(self.dataset_dir, 'targetImage','000001.tif')

        psf_stack = loadmat(PSF_PATH)  # 256, 256, 32
        Lr = tiff3Dread(LR_DATA_PATH)
        Hr = tiff3Dread(HR_DATA_PATH)

        if Lr.shape[-2:] == Hr.shape[-2:]:
            _, w, h = Hr.shape
            w = w // 8 * 8
            h = h // 8 * 8
            Hr = Hr[:, :w, :h]
            Lr = Lr[:, :w, :h]
            Lr = Lr[:, ::4, ::4]


        psf_stack = np.array(psf_stack['PSF_stack'])

        Lr = torch.from_numpy(Lr.copy()) 
        psf_stack = torch.from_numpy(psf_stack.copy())
        Hr = torch.from_numpy(Hr.copy()) 


        if len(psf_stack.shape)==3 and psf_stack.shape[1] != psf_stack.shape[2]:
            psf_stack = psf_stack.permute(2,0,1)        


        if len(psf_stack.shape)>3:
            psf_stack = psf_stack.reshape(-1,psf_stack.shape[2],psf_stack.shape[3])

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        # LF_name = self.dataset_dir
        LF_name = LR_DATA_PATH.split('\\')[-1].split('.')[0]

        rays = get_rays(Lr.shape[1]* 4, Lr.shape[2]* 4)


        if self.stack_input:
            # Lr_SAI_y = rearrange(Lr_SAI_y, '1 (u h) (v w) -> (u v) h w', u=Lr_angRes_in, v=Lr_angRes_in)
            Lr = Lr[self.weight]
            psf_stack = psf_stack[self.weight]

        rgbs = Lr.clone().float()

        data = {
            'rays' : rays,
            # 'rays_3d': rays_3d,
            'masked_input': rgbs,
            'input': rgbs,
            'psf': psf_stack,
            'gt':Hr
        }


        return data,  [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num * self.repeat


def get_rays(w, h):

    half_dx =  0.5 / w
    half_dy =  0.5 / h
    xs = torch.linspace(half_dx, 1-half_dx, w)
    ys = torch.linspace(half_dy, 1-half_dy, h)
    xv, yv = torch.meshgrid([xs, ys], indexing="ij")
    # rays = torch.stack((yv.flatten(), xv.flatten())).t()
    rays = torch.stack((yv, xv))
    rays = rays.permute(1, 2, 0)
    return rays

def get_rays_3d(w, h, z=1):

    half_dx =  0.5 / w
    half_dy =  0.5 / h
    half_dz =  0.5 / z
    xs = torch.linspace(half_dx, 1-half_dx, w)
    ys = torch.linspace(half_dy, 1-half_dy, h)
    zs = torch.linspace(half_dz, 1-half_dz, z)
    xv, yv, zv = torch.meshgrid([xs, ys, zs], indexing="ij")
    rays = torch.stack((yv, xv, zv), dim=-1)
    return rays

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def flip_SAI(data, angRes):
    if len(data.shape)==2:
        H, W = data.shape
        data = data.reshape(H, W, 1)
    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C) # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)
    return data

def gaussian_noise(image,mean,sigma):
    noise=np.random.normal(mean, sigma, image.shape)
    gaussian_out=image+noise.astype(np.float32)
    gaussian_out=np.clip(gaussian_out,0,1)
    return gaussian_out


def augmentation_hr(data):
    if random.random() < 0.5:
        data = np.flip(data, axis=0)
    if random.random() < 0.5:
        data = np.flip(data, axis=1)
    if random.random() < 0.5:
        data = np.rot90(data, -1)
    if random.random() < 0.5:
        data = np.rot90(data, -1)   
    if random.random() < 0.5:
        data = np.rot90(data, -1)      
    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label


def augmentation_lfh(data, label, psf):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
        psf = psf[:,:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
        psf = psf[:,::-1]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
        psf = psf.transpose(0, 2, 1)
    return data, label, psf