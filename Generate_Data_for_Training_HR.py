import argparse
import os
import h5py
from utils.imresize import *
from pathlib import Path
import scipy.io as scio
import sys
from utils.utils import rgb2gray, loadmat, psf2otf, ZeroPad_np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angRes", type=int, default=4, help="angular resolution")
    parser.add_argument("--scale_factor", type=int, default=2, help="4, 2")
    parser.add_argument('--data_for', type=str, default='training1', help='')
    parser.add_argument('--src_data_path', type=str, default=r'/run/zx/code/dataset', help='')
    parser.add_argument('--save_data_path', type=str, default=r'/run/zx/dataset/data', help='')

    args = parser.parse_args()
    # args.src_data_path = r'/root/user/lfh/dataset/source/Flickr2K'
    # args.save_data_path = r'/root/user/lfh/dataset/target_hr/data'

    return args


def main(args):
    angRes, scale_factor = args.angRes, args.scale_factor
    patchsize_lr = 32
    patchsize_hr =  patchsize_lr * scale_factor
    stride = patchsize_hr 

    ''' dir '''
    save_dir = Path(args.save_data_path + '_for_' + args.data_for)
    save_dir.mkdir(exist_ok=True)

    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()

    
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['DIV2K_train_HR']:
            continue

        idx_save = 0
        name_dataset = src_datasets[index_dataset]
        sub_save_dir = save_dir.joinpath(name_dataset)
        sub_save_dir.mkdir(exist_ok=True)

        hr_dataset = args.src_data_path + '/' + name_dataset + '/'

        for root, dirs, files in os.walk(hr_dataset):
            files.sort()
            for file in files:
                idx_scene_save = 0
                print('Generating training data of Scene_%s in Dataset %s......\t' %(file, name_dataset))

                hr_rgb = np.array(Image.open(hr_dataset+file)) # h, w
                hr_rgb = hr_rgb/255.0
                (H, W, C) = hr_rgb.shape
                hr_gray = rgb2gray(hr_rgb)

                U = angRes
                V = angRes

                for h in range(0, H - patchsize_hr + 1, stride):
                    for w in range(0, W - patchsize_hr + 1, stride):
                        
                        idx_save = idx_save + 1
                        idx_scene_save = idx_scene_save + 1
                        Hr_SAI_y = np.zeros((1 * patchsize_hr, 1 * patchsize_hr),dtype='single')
                        Hr_SAI_y = hr_gray[h: h + patchsize_hr, w: w + patchsize_hr]
                        file_name = [str(sub_save_dir) + '/' + '%06d'%idx_save + '.h5']
                        with h5py.File(file_name[0], 'w') as hf:
                            hf.create_dataset('hr_center', data=Hr_SAI_y, dtype='single')
                            hf.close()
                            pass
                        pass
                    pass
                print('%d training samples have been generated\n' % (idx_scene_save))
                pass
            pass
        pass

    pass



if __name__ == '__main__':
    args = parse_args()

    main(args)

