import argparse
import os
import h5py
from utils.imresize import *
from pathlib import Path
import scipy.io as scio
import sys
from utils.utils import rgb2gray, loadmat, simulation, load_psf
from PIL import Image
import imageio
import cv2
import scipy
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angRes", type=int, default=8, help="angular resolution")
    parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")
    parser.add_argument('--data_for', type=str, default='test', help='')
    parser.add_argument('--src_data_path', type=str, default=r'', help='')
    parser.add_argument('--save_data_path', type=str, default=r'', help='')
    parser.add_argument('--psf_path', type=str, default=r'.mat', help='')

    args = parser.parse_args() 
    args.src_data_path = r'/root/user/lfh/dataset/source'
    args.save_data_path = r'/root/user/lfh/dataset/target/data'
    args.psf_name = r'3'
    args.psf_path = r'/root/user/lfh/dataset/train_psf/psf_simulation_defocus/'+str(args.angRes)+'x'+str(args.angRes)+'/' + args.psf_name + '.mat'
    return args


def main(args):
    print(args.psf_path)
    angRes, scale_factor = args.angRes, args.scale_factor

    ''' dir '''
    save_dir = Path(args.save_data_path + '_for_' + args.data_for)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath('SR_' + str(angRes) + 'x' + str(angRes) + '_' + str(scale_factor) + 'x')
    save_dir.mkdir(exist_ok=True)

    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()

 
    ''' load the light field point spread function'''
    psf_name = (args.psf_path).split('/')[-1].split('.')[0]
    psf_stack = load_psf(args.psf_path, angRes, 1)


    '''generate the light field image'''
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['Set5']:
            continue

        idx_save = 0
        name_dataset = src_datasets[index_dataset]
        sub_save_dir = save_dir.joinpath(name_dataset+'_' + psf_name)

        sub_save_dir.mkdir(exist_ok=True)

        hr_dataset = args.src_data_path + '/' + name_dataset + '/'

        for root, dirs, files in os.walk(hr_dataset):
            files.sort()
            for file in files:
                print('Generating Test data of Scene_%s in Dataset %s......\t' %(file, name_dataset))

                hr_rgb = np.array(Image.open(hr_dataset+file)) # h, w
                hr_rgb = hr_rgb / 255.0
                (H, W, C) = hr_rgb.shape
                hr_gray = rgb2gray(hr_rgb)

                H = H // angRes * angRes
                W = W // angRes * angRes
                print(H, W)

                hr_gray = hr_gray[0:H, 0:W]


                # # cv2 version
                # Hr_SAI_y_out = []
                # for i in range(angRes * angRes):
                #     tmp_psf = psf_stack[i]
                #     # cv2.filter2D 为相关运算， kernel 需要旋转180度
                #     tmp_psf = tmp_psf[::-1, ...][:, ::-1]
                #     hr = cv2.filter2D(hr_gray.copy(), -1, tmp_psf)
                #     Hr_SAI_y_out.append(hr)
                # Hr_SAI_y_out = np.stack(Hr_SAI_y_out)
                # Lr_SAI_y_out = Hr_SAI_y_out[:,::scale_factor,::scale_factor]
                # Lr_SAI_y_out = np.clip(Lr_SAI_y_out, 0, 1)

                # torch version
                hr_gray_torch = torch.from_numpy(hr_gray).unsqueeze(0).unsqueeze(0)
                psf_stack_torch = torch.from_numpy(psf_stack).unsqueeze(0).type(torch.FloatTensor)
                Lr_SAI_y_out = simulation(hr_gray_torch, psf_stack_torch, stride=scale_factor)
                Lr_SAI_y_out = Lr_SAI_y_out.numpy().squeeze(0)
                Lr_SAI_y_out = np.clip(Lr_SAI_y_out, 0, 1)


                file_name = [str(sub_save_dir) + '/' + '%06d'%idx_save + '.h5']
                with h5py.File(file_name[0], 'w') as hf:
                    hf.create_dataset('Lr_SAI_y', data=Lr_SAI_y_out, dtype='single')
                    hf.create_dataset('Hr_SAI_y', data=hr_gray, dtype='single')
                    hf.create_dataset('psf_stack', data=psf_stack, dtype='single')
                    hf.close()
                    pass
                pass
                idx_save = idx_save + 1 
                print('%d test samples have been generated\n' % (idx_save))
            pass
        pass

    pass



if __name__ == '__main__':
    args = parse_args()

    main(args)

