import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='SR', help='SR, RE')

# LF_SR
parser.add_argument("--angRes", type=int, default=4, help="angular resolution")
parser.add_argument("--scale_factor", type=int, default=2, help="8, 4, 2")

parser.add_argument('--model_name', type=str, default='LFT', help="model name")
parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")
parser.add_argument("--path_pre_pth", type=str, default='/run/zx/dataset/SR_4x4_2x_11-13/ALL/RepLFSR/checkpoints/epoch_60_model.pth', help="path for pre model ckpt") #'/run/zx-dm/dataset/SR_4x4_2x_09_16/ALL/RepLFSR/checkpoints/best_model.pth'
parser.add_argument('--data_name', type=str, default='ALL',
                    help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL(of Five Datasets)')
parser.add_argument('--path_for_train', type=str, default='/run/zx/dataset/data_for_training1/')
parser.add_argument('--path_for_test', type=str, default='/run/zx/dataset/data_for_test/')
parser.add_argument('--path_log', type=str, default='log/')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
parser.add_argument('--n_steps', type=int, default=5, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.8, help='gamma')
parser.add_argument('--epoch', default=200, type=int, help='Epoch to run [default: 50]')

parser.add_argument('--device', type=str, default='cuda:7')
parser.add_argument('--num_workers', type=int, default=8, help='num workers of the Data Loader')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0, )

parser.add_argument('--noise_std_max', type=float, default=25, help='noise_std_max')
parser.add_argument('--train_data_num', type=int, default=-1, help='train_data_num')
parser.add_argument('--mark', type=str, default='-08-test', help='mark')
parser.add_argument('--stack_input', type=bool, default=False, help='the formate of input')

parser.add_argument('--view_shuffle', type=bool, default=False, help='')

parser.add_argument('--resize', type=bool, default=False, help='')

parser.add_argument('--channels', type=int, default=64, help='channels')
parser.add_argument('--n_group', type=int, default=4, help='n_group')
parser.add_argument('--n_block', type=int, default=4, help='n_block')
parser.add_argument('--data_range', type=int, default=1, help='data range')

parser.add_argument('--repeat', type=int, default=1, help='repeat')

parser.add_argument('--angnum', type=int, default=9, help='angnum')

parser.add_argument('--path_for_single', type=str, default='', help='')

parser.add_argument('--mixed_prob', type=int, default=0, help='')

parser.add_argument('--psf_path', type=str, default='psf/', help='')

parser.add_argument('--psf_name', type=str, default='ALL', help='')

parser.add_argument('--test_data_name', type=str, default='test_image_08/rggb_all/')

parser.add_argument('--conbines_test_images',type=str, default = 'combined_image/08/res/')

parser.add_argument('--test_epoch', type=str, default='1')

parser.add_argument('--psf_downsample', type=int, default='2')

parser.add_argument('--early_stop', type = str, default = False, help = 'is early stop the trainning ?')
parser.add_argument('--patience', type = int, default = 20, help = 'The number step of stop trainning')


args = parser.parse_args()


# args.psf_name = '-9,-6,-3,0,3,6,9'

if args.task == 'SR':
    args.angRes_in = args.angRes
    args.angRes_out = 1
    args.patch_size_for_test = 32
    args.stride_for_test = 16
    args.minibatch_for_test = 1

    # args.path_for_train = '/root/user/lfh/dataset/target_hr/data_for_training/'
    # args.path_for_train = '/home/wangqian/datasets/data_fundus/_for_training/'
    
    # args.path_for_test = '/run/zx-dm/dataset/data_for_test/'
    args.path_log = '/run/zx/dataset/'
    args.data_name = 'ALL'

# args.path_pre_pth = 'D:\workspace\experiments\UnsupervisedLFSR\log\SR_8x8_4x_backbone_12800_upsample\ALL\RepLFSR_v1\RepLFSR_v1_8x8_4x_epoch_10_model.pth'

if 1:
    weight = [True]*args.angRes_in*args.angRes_in
    center = int(args.angRes_in/2)
    angnum = args.angnum
    if angnum > 0:
        count = 0
        for u in range(args.angRes_in):
            for v in range(args.angRes_in):
                if ((u-center+0.5)**2 +(v-center+0.5)**2 > angnum):
                    weight[u*args.angRes_in+v] = False
                    count = count + 1
            
    args.weight = weight
    args.in_channels = args.angRes_in*args.angRes_in - count


if args.task == 'Single_SR':
    args.angRes_in = args.angRes
    args.angRes_out = 1
    args.patch_size_for_test = 64
    args.stride_for_test = 32
    args.minibatch_for_test = 1




