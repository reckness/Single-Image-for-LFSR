import os
if __name__ == '__main__':
    
  
    model_name = 'RCAN_small'

    test_data_name = 'test_single_data'

    angRes = 8
    scale_factor = 4
    mark = 'psf_real_f4_new_all_psf_8x8x4_scan'
    angnum = 9

    ROOT_PATH = '/root/user/lfh/experiments/UnsupervisedLFSR/log/SR_{}x{}_{}x_{}/ALL/{}/checkpoints'.format(angRes,angRes,scale_factor, mark, model_name)

    for epoch in range(27, 28, 1):
        path_pre_pth = '{}/epoch_{:0>2d}_model.pth'.format(ROOT_PATH, epoch)
        cmd = 'python inference_stack.py --model_name {} --angRes {} --scale_factor {} --path_pre_pth {} --test_epoch {} --mark {} --test_data_name {} --angnum {}'.format(model_name, angRes, scale_factor, path_pre_pth, epoch, mark, test_data_name, angnum)
        os.system(cmd)  


