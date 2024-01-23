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

def load_model(args):
      
    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device) 

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


    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of parameters: %.4fM' % (total / 1e6))

    return net

def inference_img(args, net, DATA_PATH, OUTPUT_PATH):

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    DATA_RANGE=255

    Num = args.angRes


    IMG_PATH = os.path.join(DATA_PATH, 'view_1.tif')
    img = cv2.imread(IMG_PATH)

    PATCH_SIZE_H, PATCH_SIZE_W, _ = img.shape
    # PATCH_SIZE_H, PATCH_SIZE_W = 100, 100

    test_data = np.zeros([PATCH_SIZE_H*Num, PATCH_SIZE_W*Num])

    for u in range(0, Num):
        for v in range(0, Num):
            IMG_PATH = os.path.join(DATA_PATH, 'view_'+ str(u*Num+v)+'.tif')
            img = cv2.imread(IMG_PATH)
            test_data[u*PATCH_SIZE_H:(u+1)*PATCH_SIZE_H,v*PATCH_SIZE_W:(v+1)*PATCH_SIZE_W] = img[:PATCH_SIZE_H,:PATCH_SIZE_W,0]

    # test_data = test_data/DATA_RANGE
    test_data = torch.from_numpy(test_data).cuda().type(torch.float32)

    subLFin = LFdivide(test_data, args.angRes_in, args.patch_size_for_test, args.stride_for_test)


    numU, numV, H, W = subLFin.size()
    subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
    subLFout = torch.zeros(numU * numV, 1, args.angRes_out * args.patch_size_for_test * args.scale_factor,
                            args.angRes_out * args.patch_size_for_test * args.scale_factor)

    ''' SR the Patches '''
    for i in range(0, numU * numV, args.minibatch_for_test):
        tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
        with torch.no_grad():
            net.eval()
            torch.cuda.empty_cache()
            out = net(tmp.to(device))
            subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out
    subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

    ''' Restore the Patches to LFs '''
    Sr_4D_y = LFintegrate(subLFout, args.angRes_out, args.patch_size_for_test * args.scale_factor,
                            args.stride_for_test * args.scale_factor, numU*numV*H*6,  numU*numV*W*6)
    output = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

    Sr_4D_y = Sr_4D_y.cpu().detach().numpy()
    output = output.squeeze()
    output = output.cpu().detach().numpy()


    for u in range(args.angRes_out):
        for v in range(args.angRes_out):
            img = Sr_4D_y[u, v, :, :]
            path = os.path.join(OUTPUT_PATH, str(args.model_name) + '_'+str(args.mark)+ '_view_'+str(u*Num+v)+'.tif')
            cv2.imwrite(path, img)


def inference_main(args, net):
    from option import args
    import time
    net = load_model(args)

    DATA_PATH = '/cpfs01/user/aiforscience2/DATA/LFSR/real_data/4x4/'
    OUTPUT_PATH = '/cpfs01/user/aiforscience2/lfh/experiments/BasicLFSR/real_data_result'

    input_path_list = os.listdir(DATA_PATH)


    for path in input_path_list:

        if path[0] == '.':
            continue
        start_time = time.time()
    
        input_path = os.path.join(DATA_PATH, path)
        out_path = os.path.join(OUTPUT_PATH, args.model_name, path)
        inference_img(args, net, input_path, out_path)
        end_time = time.time()

        print("Inference Image {}, time: {}".format(path, (end_time-start_time)))


if __name__ == '__main__':
    from option import args
    import time

    net = load_model(args)

    DATA_PATH = '/cpfs01/user/aiforscience2/DATA/LFSR/real_data/4x4/'
    OUTPUT_PATH = '/cpfs01/user/aiforscience2/lfh/experiments/BasicLFSR/real_data_result'

    input_path_list = os.listdir(DATA_PATH)


    for path in input_path_list:

        if path[0] == '.':
            continue
        start_time = time.time()
    
        input_path = os.path.join(DATA_PATH, path)
        out_path = os.path.join(OUTPUT_PATH, args.model_name, path)
        inference_img(args, net, input_path, out_path)
        end_time = time.time()

        print("Inference Image {}, time: {}".format(path, (end_time-start_time)))




