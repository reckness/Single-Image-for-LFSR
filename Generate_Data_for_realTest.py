import argparse
import os
import h5py
from utils.imresize import *
from pathlib import Path
import scipy.io as scio
import sys
from utils.utils import rgb2gray, loadmat, psf2otf, ZeroPad_np,pre_img
from PIL import Image
import imageio
import cv2
import scipy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angRes", type=int, default=4, help="angular resolution")
    parser.add_argument("--scale_factor", type=int, default=2, help="4, 2")
    parser.add_argument('--data_for', type=str, default='realTest', help='')
    parser.add_argument('--src_data_path', type=str, default=r'/run/zx/code/combined_image/08/res', help='')
    parser.add_argument('--save_data_path', type=str, default=r'/run/zx/dataset/data', help='')
    parser.add_argument('--psf_path', type=str, default=r'.mat', help='')

    args = parser.parse_args() 
    return args


def main(args):
    print(args.psf_path)
    angRes, scale_factor = args.angRes, args.scale_factor

    ''' dir '''
    save_dir = Path(args.save_data_path + '_for_' + args.data_for)
    save_dir.mkdir(exist_ok=True)
    
    real_dataset = args.src_data_path + '/'

    for root, dirs, files in os.walk(real_dataset):
        files.sort()
        for file in files:
            print('Generating real data of Scene_%s .....\t' % file)

            hr_rgb = np.array(Image.open(real_dataset+file)) # h, w
            hr_rgb = hr_rgb / 255.0
            (H, W, C) = hr_rgb.shape
            hr_gray = rgb2gray(hr_rgb)

            H = H // angRes * angRes
            W = W // angRes * angRes
            print(H, W)

            hr_gray = hr_gray[0:H, 0:W]

            Hr_SAI_y_out = []
            for i in range(angRes * angRes):
                tmp_psf = psf_stack[i]
                # cv2.filter2D 为相关运算， kernel 需要旋转180度
                tmp_psf = tmp_psf[::-1, ...][:, ::-1]
                hr = cv2.filter2D(hr_gray.copy(), -1, tmp_psf)
                Hr_SAI_y_out.append(hr)


            Hr_SAI_y_out = np.stack(Hr_SAI_y_out)
            # print('Hr',Hr_SAI_y_out.shape)
            # Hr_SAI_y = pre_img(Hr_SAI_y_out,angRes)
            Hr_SAI_y_out = np.clip(Hr_SAI_y_out, 0, 1)
            Lr_SAI_y_out = Hr_SAI_y_out[:,::scale_factor,::scale_factor]
            # Lr_SAI_y_out = pre_img(Lr_SAI_y_out,angRes)
            # print("HR:",Hr_SAI_y.shape)
            # print('LR:',Lr_SAI_y_out.shape)


            file_name = [str(sub_save_dir) + '/' + '%06d'%idx_save + '.h5']
            with h5py.File(file_name[0], 'w') as hf:
                hf.create_dataset('Lr_SAI_y', data=Lr_SAI_y_out, dtype='single')
                hf.create_dataset('Hr_SAI_y', data=Hr_SAI_y_out, dtype='single')
                hf.create_dataset('psf_stack', data=psf_stack, dtype='single')
                hf.close()
                pass
            pass
            idx_save = idx_save + 1 
            print('%d test samples have been generated\n' % (idx_save))
        pass
    pass

   



if __name__ == '__main__':
    args = parse_args()

    main(args)

