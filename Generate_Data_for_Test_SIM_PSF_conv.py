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
    parser.add_argument('--data_for', type=str, default='test1', help='')
    parser.add_argument('--src_data_path', type=str, default=r'/run/zx/code/dataset', help='')
    parser.add_argument('--save_data_path', type=str, default=r'/run/zx/dataset/data', help='')
    parser.add_argument('--psf_path', type=str, default=r'.mat', help='')

    args = parser.parse_args() 
    # args.src_data_path = r'/root/user/lfh/dataset/source'
    # args.save_data_path = r'/root/user/lfh/dataset/target/data'
    args.psf_name = r'20'
    args.psf_path = r'/run/zx/code/psf/'+ args.psf_name + '_psf_pupil4_layer_1'+ '.mat'

    return args


def main(args):
    print(args.psf_path)
    angRes, scale_factor = args.angRes, args.scale_factor

    ''' dir '''
    save_dir = Path(args.save_data_path + '_for_' + args.data_for)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath('SR_' + str(angRes) + 'x' + str(angRes) + '_' + str(scale_factor) + 'x_08')
    save_dir.mkdir(exist_ok=True)

    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()

 
    ''' load the light field point spread function'''
    psf_name = (args.psf_path).split('/')[-1].split('.')[0]
    psf_stack = loadmat(args.psf_path)
    if 'PSF_stack' in psf_stack.keys():
        psf_stack = psf_stack['PSF_stack']
    elif 'psf_z' in psf_stack.keys():
        psf_stack = psf_stack['psf_z']
    else:
        psf_stack = psf_stack['psf']['psf_z'][0][0]

    psf_stack = np.array(psf_stack)

    tmp = []
    if psf_stack.shape[0] == angRes:
        for u in range(angRes):
            for v in range(angRes):
                tmp.append(psf_stack[u, v, :, :])
    elif psf_stack.shape[2] == angRes:
        for u in range(angRes):
            for v in range(angRes):
                tmp.append(psf_stack[:, :, u, v])
    else:
        return 0 

    ''' psf sum normalization'''
    psf_stack = np.stack(tmp)
    tmp = np.sum(psf_stack[angRes*(angRes//2-1)+angRes//2-1,:,:])
    # for u in range(angRes*angRes):
    #     psf_stack[u,:,:] = psf_stack[u,:,:]/tmp


    for u in range(angRes*angRes):
        psf_stack[u,:,:] = psf_stack[u,:,:]/np.sum(psf_stack[u,:,:])


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

    pass



if __name__ == '__main__':
    args = parse_args()

    main(args)

