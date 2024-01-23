import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import importlib
import torch
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import MultiTestSetDataLoader
from collections import OrderedDict
from train import test
import torch.onnx
from torch.autograd import Variable

import cv2
import numpy as np


def main(args):
    ''' Create Dir for Save '''
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    # ''' DATA TEST LOADING '''

    # DATA_PATH = 'D:\workspace\Aberration_correction_metasensor88\Data\sub_image\/20221229\/22/img22.tif'

    # image = cv2.imread(DATA_PATH)
    # image = np.expand_dims(image[:,:,0],[0,1])
    # image = image/255


    # cv2.imwrite('lr.png', image.reshape(64,64,1)*255)
    # image = torch.from_numpy(image).cuda().type(torch.float32)

    # import ipdb
    # ipdb.set_trace()

    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
    else:
        ckpt_path = args.path_pre_pth
        print(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.' + k  # add `module.`
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
            pass
        pass

    net = net.to(device)
    cudnn.benchmark = True
    dummy_input = Variable(torch.randn(1, 32, 64, 64)).cuda()

    torch.onnx.export(net, dummy_input, "RCAN_net_8x8_4x_64.onnx", opset_version=10, verbose=True)



if __name__ == '__main__':
    from option import args

    main(args)


