import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import importlib
import torch
import torch.backends.cudnn as cudnn
from utils.utils import *
from collections import OrderedDict
import cv2
import numpy as np
from utils.imresize import *
import tifffile as tiff


def load_model(args):      
    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device) 

    ''' MODEL LOADING '''
    print('\n Model Initial ...')
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

def inference_img(args, net, DATA_PATH, OUTPUT_PATH, angnum):

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    DATA_RANGE=args.data_range
    Num = args.angRes

    test_data = tiff.imread(DATA_PATH)
    print(test_data.shape)
    # test_data = np.expand_dims(test_data,axis = 0)
    # test_data = test_data.reshape(16,64,84)
    # print(test_data.shape)

    # PATCH_SIZE_H, PATCH_SIZE_W = args.patch_size_h, args.patch_size_w

    # import ipdb
    # ipdb.set_trace()

    # filenames = os.listdir(DATA_PATH)
    # if 'view_0.tif' in filenames:
    #     IMG_PATH = os.path.join(DATA_PATH, 'view_0.tif')
    # elif 'view_001.tif' in filenames:
    #     IMG_PATH = os.path.join(DATA_PATH, 'view_001.tif')
    # else:
    #     return 0

    # img = cv2.imread(IMG_PATH)

    if args.patch_size_h > 0:
        OFFSET_X, OFFSET_Y = args.offset_x, args.offset_y 
        PATCH_SIZE_H, PATCH_SIZE_W = args.patch_size_h, args.patch_size_w
    else:
        OFFSET_X, OFFSET_Y = 0, 0
        _, PATCH_SIZE_H, PATCH_SIZE_W = test_data.shape 

    # test_data = np.zeros([Num*Num, PATCH_SIZE_H, PATCH_SIZE_W])

    # for u in range(0, Num):
    #     for v in range(0, Num):
    #         if 'view_0.tif' in filenames:
    #             IMG_PATH = os.path.join(DATA_PATH, 'view_'+ str(u*Num+v)+'.tif')
    #         elif 'view_001.tif' in filenames:
    #             IMG_PATH = os.path.join(DATA_PATH, 'view_'+ "{:0>3d}".format(u*Num+v+1)+'.tif')
    #         else:
    #             return 0
    #         img = cv2.imread(IMG_PATH)
    #         if len(img.shape) == 3:
    #             img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         img = np.array(img).astype('float')
    #         img = img/np.max(img)
    #         test_data[u*Num+v, :,:] = img[OFFSET_X:OFFSET_X+PATCH_SIZE_H, OFFSET_Y:OFFSET_Y+PATCH_SIZE_W]

    test_data = test_data/np.max(test_data)
    # test_data = test_data/255.0

    ### save low resolution image
    lr = test_data[ Num * (Num//2-1) + (Num//2-1),:,:]
    img = (lr*255).astype('uint8')
    path = os.path.join(OUTPUT_PATH, str(args.model_name) + '_'+'LR.bmp')
    cv2.imwrite(path, img)


    ### save bicubic image
    img = imresize(img, scalar_scale=args.scale_factor)
    path = os.path.join(OUTPUT_PATH, str(args.model_name) + '_'+'Bic.bmp')
    cv2.imwrite(path, img)

    weight = [True]*Num*Num
    center = int(Num/2)
    if angnum > 0:
        count = 0
        for u in range(args.angRes_in):
            for v in range(args.angRes_in):
                if ((u-center+0.5)**2 +(v-center+0.5)**2 > angnum):
                    weight[u*args.angRes_in+v] = False
            

    test_data = test_data[weight]
    # print('Test:',test_data.shape)

    test_data = torch.from_numpy(test_data).cuda().type(torch.float32).unsqueeze(0)

    subLFin, numU, numV = Stackdivide(test_data, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
    num, views, H, W = subLFin.size()
    subLFin = pre_img_LF(subLFin,4)
    print('dd',subLFin.shape)
    subLFout = torch.zeros(num, 1, args.angRes_out * args.patch_size_for_test * args.scale_factor,
                        args.angRes_out * args.patch_size_for_test * args.scale_factor)

    tmp_input_data = {}
    tmp_input_data['psf'] = 0

    ''' SR the Patches '''
    for i in range(0, num, args.minibatch_for_test):
        tmp = subLFin[i:min(i + args.minibatch_for_test, num), :, :, :]

        with torch.no_grad():
            net.eval()
            torch.cuda.empty_cache()
            tmp_input_data['input'] = tmp
            out = net(tmp_input_data)
            if isinstance(out, dict):
                sr = out['sr']
            else:
                sr = out

            sr  = conbine_img_Lf(sr,4)
            print('sr',sr.shape)
            subLFout[i:min(i + args.minibatch_for_test, num), :, :, :] = sr

    
    subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

    ''' Restore the Patches to LFs '''
    Sr_4D_y = LFintegrate(subLFout, args.angRes_out, args.patch_size_for_test * args.scale_factor,
                        args.stride_for_test * args.scale_factor, PATCH_SIZE_H*args.scale_factor, PATCH_SIZE_W*args.scale_factor)


    Sr_4D_y = Sr_4D_y.cpu().detach().numpy()
    Sr_4D_y = Sr_4D_y.clip(0,1)

    for u in range(args.angRes_out):
        for v in range(args.angRes_out):
            img = Sr_4D_y[u, v, :, :]
            path = os.path.join(OUTPUT_PATH, str(args.model_name) + '_'+str(args.test_epoch)+'.bmp')
            cv2.imwrite(path, img*255)





def pre_img_LF (x,angRes):

    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)
    b,c,h,w = x.shape
    # print(x.shape)
    output_img = torch.empty(3960,1,h*angRes,w*angRes).to(device)
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


def conbine_img_Lf(x,angRes):
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)
    b,c,h,w = (x.shape)
    # print('xaxa',x.shape)
    output_img = torch.empty(1,16,64,64).to(device)
    # print(output_img.shape)
    sub_blocks = []
    for i in range(4):
        for j in range(4):
            sub_block = x[:, :, i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
            sub_blocks.append(sub_block)
    
    # output_img = torch.cat(sub_blocks,dim = 1)
    output_img = x[ : ,: ,64*1:64*2,64*1:64*2] 
    # print('111',output_img.shape)
    return output_img



        

if __name__ == '__main__':
    from option import args
    import time

    args.data_range = 1
    args.offset_x = 304
    args.offset_y = 502
    args.patch_size_h = -1 #64
    args.patch_size_w = 64
    

    angnum = args.angnum

    DATASET_NAME = args.test_data_name

    DATA_PATH = r'/run/zx/code'+'/'+DATASET_NAME +'/'       #+str(args.angRes_in) + 'x' + str(args.angRes_in)+'/'
    OUTPUT_PATH = r'/run/zx/dataset' + '/' + DATASET_NAME + '/' +str(args.angRes_in) + 'x' + str(args.angRes_in)+'/'

    print('test real data')

    net = load_model(args)

    conbines(DATA_PATH,args.conbines_test_images,args.angRes)

    input_path_list = os.listdir(args.conbines_test_images)

    for path in input_path_list:

        if path[0] == '.':
            continue
        start_time = time.time()
    
        input_path = os.path.join(args.conbines_test_images, path)
        out_path = os.path.join(OUTPUT_PATH, args.mark, args.model_name, path)
        inference_img(args, net, input_path, out_path, angnum)
        end_time = time.time()

        print("Inference Image {}, time: {}".format(path, (end_time-start_time)))




