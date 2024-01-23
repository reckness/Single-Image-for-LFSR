import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TestRealSingleDataLoader
from collections import OrderedDict
import imageio
import cv2


def main(args):
    ''' Create Dir for Save'''
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    logger.log_string('\nWorking dirtory is ' + str(log_dir))



    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    test_Loader = TestRealSingleDataLoader(args)
    # logger.log_string("The number of validation data is: %d" % length_of_tests)
    test_Loader = DataLoader(dataset=test_Loader, num_workers=args.num_workers, batch_size=1, shuffle=False)

    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    total = sum([param.nelement() for param in net.parameters()])
    logger.log_string('\nNumber of parameters: %.4fM' % (total / 1e6))



    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
        start_epoch = 0
        logger.log_string('Do not use pre-trained model!')
    else:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            start_epoch = checkpoint['epoch']
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k  # add `module.`
                    new_state_dict[name] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
            except:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
        except:
            net = MODEL.get_model(args)
            net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass
    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)


    ''' LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)


    ''' Optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)


    ''' TRAINING & TEST '''
    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

        ''' Validation '''
        step = 1
        if (idx_epoch + 1)%step==0 or idx_epoch > args.epoch-step:
            # with torch.no_grad():
                ''' Create Excel for PSNR/SSIM '''
                excel_file = ExcelFile()

                psnr_testset = []
                ssim_testset = []
                test_name = args.path_for_single.split('/')[-1]

                # for index, test_name in enumerate(test_Names):
                    # test_loader = test_Loaders[index]
                epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
                epoch_dir.mkdir(exist_ok=True)
                # save_dir = epoch_dir.joinpath(test_name)
                # save_dir.mkdir(exist_ok=True)
                save_dir = epoch_dir

                psnr_iter_test, ssim_iter_test, LF_name, loss_iter_test = test(test_Loader, device, net, criterion, optimizer, save_dir)

                excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)

                psnr_epoch_test = float(np.array(psnr_iter_test)[-1])
                ssim_epoch_test = float(np.array(ssim_iter_test)[-1])
                loss_iter_test = float(np.array(loss_iter_test)[-1])


                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                logger.log_string('The %dth Test on %s, loss is %.3f psnr/ssim is %.2f/%.3f' % (
                idx_epoch + 1, test_name, loss_iter_test, psnr_epoch_test, ssim_epoch_test))
                pass
                # psnr_mean_test = float(np.array(psnr_testset).mean())
                # ssim_mean_test = float(np.array(ssim_testset).mean())
                # logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.5f'
                #                   % (psnr_mean_test, ssim_mean_test))
                # excel_file.xlsx_file.save(str(epoch_dir) + '/'+ str(test_name)+ '/evaluation.xls')
                # pass
            # pass

        ''' Save PTH  '''
        if args.local_rank == 0:
            save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
            args.model_name, args.angRes_in, args.angRes_in, args.scale_factor, idx_epoch + 1)
            state = {
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
            }
            torch.save(state, save_ckpt_path)
            logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))

        ''' scheduler '''
        scheduler.step()
        pass
    pass




def test(test_loader, device, net, criterion, optimizer, save_dir=None):
    LF_iter_test = []
    loss_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []

    pbar =  tqdm(enumerate(test_loader), total=len(test_loader), ncols=100)
    for idx_iter, (input_data, data_info, LF_name) in pbar:

        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()
     
        for key in input_data.keys():
            input_data[key] = input_data[key].to(device) 

        output = net(input_data, data_info)
        loss = criterion(input_data, output, data_info)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        loss = float(np.array(loss.data.cpu()))
        ''' Save Gray '''


        loss_iter_test.append(loss)
        # label = input_data['gt'][:,:,12:-12,12:-12]


        # if Sr.shape[1]>1:
        #     Sr = torch.mean(Sr, dim=1, keepdim=True) 
        # Sr = Sr[:,:,12:-12,12:-12]
        # Sr[Sr>1] =1
        # Sr[Sr<0] =0

        # ''' Calculate the PSNR & SSIM '''
        # psnr, ssim = cal_metrics(args, Sr, label)
        Sr = output['sr']
        psnr, ssim = 0, 0

        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)        
        LF_iter_test.append(LF_name[0])
        pbar.set_postfix(psnr= '{:.2f}'.format(psnr), ssim= '{:.3f}'.format(ssim), loss = '{:.4f}'.format(loss))

        step = 50
        if save_dir is not None  and ((idx_iter+1) % step==0):
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            img = (Sr.squeeze(0).squeeze(0).cpu().detach().numpy())
            img[img<0] = 0
            img = img/np.max(img)*255
            img = img.astype('uint8')

            
            path = str(save_dir_) + '/' + str((idx_iter+1)) + '_' + LF_name[0] + '.bmp'
            imageio.imwrite(path, img)

        pass


    return psnr_iter_test, ssim_iter_test, LF_iter_test, loss_iter_test

if __name__ == '__main__':
    from option import args

    main(args)
