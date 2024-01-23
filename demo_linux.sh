

# 20230410 
# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_un_transpose --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 0 --train_data_num 5120 --channels 64 --n_group 8 --n_block 1 --mark RCAN --stack_input True  --lr 2e-4

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_un_transpose_large_v2 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 0 --train_data_num 5120 --channels 64 --n_group 8 --n_block 1 --mark RCAN --stack_input True  --lr 2e-4

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_un_transpose_large_v2 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 0 --train_data_num 5120 --channels 64 --n_group 8 --n_block 1 --mark RCAN_multi --stack_input True  --lr 2e-4

# python inference_stack.py --model_name RCAN_bic_large --angRes 8 --scale_factor 4 --path_pre_pth '/root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN/ALL/RCAN_bic_large/checkpoints/epoch_18_model.pth' --mark 18


# # 20230411

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_un_transpose_large_v2 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 0 --train_data_num 5120 --channels 64 --n_group 8 --n_block 1 --mark RCAN_3D --stack_input True  --lr 2e-4


# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_un_bic_3D --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 0 --train_data_num 10000 --channels 64 --n_group 8 --n_block 1 --mark RCAN_3D --stack_input True  --lr 2e-4

# python inference_stack.py --model_name RCAN_un_transpose_large_v2 --angRes 8 --scale_factor 4 --path_pre_pth '/root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_multi/DIV2K_train_HR_psf_-1,DIV2K_train_HR_psf_1,DIV2K_train_HR_psf_wo_aberration/RCAN_un_transpose_large_v2\checkpoints/epoch_77_model.pth' --mark 77


### 20230413


#  ablation study, compare the supervised and unsupervised method for noise 
####
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_bic_large --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 10 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4

# 梯度爆炸
# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_un_bic_large --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 10 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4

# 梯度爆炸
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_un_bic_3D --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 50 --train_data_num 20000  --mark RCAN_3D --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_3D/ALL/RCAN_un_bic_3D/checkpoints/epoch_27_model.pth

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_pixel_shuffle --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 10 --train_data_num 20000  --mark RCAN_3D --stack_input True  --lr 2e-4
# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_un_pixel_shuffle --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 10 --train_data_num 20000  --mark RCAN_3D --stack_input True  --lr 2e-4

# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_bic_deconv_v2 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 10 --train_data_num 20000  --mark RCAN_3D --stack_input True  --lr 2e-4

# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_bic_deconv --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 10 --train_data_num 20000  --mark RCAN_3D --stack_input True  --lr 2e-4

### 20230414
# 梯度爆炸
# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_un_bic_large_50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 50 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4
# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_un_bic_large_100_fft --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_pixel_shuffle_deconv_large --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4


# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_un_bic_large_50_fft --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 50 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4

# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_un_bic_large_25 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 50 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4
# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_un_pixel_shuffle --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 50 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_3D/ALL/RCAN_un_pixel_shuffle/checkpoints/epoch_27_model.pth


# CUDA_VISIBLE_DEVICES=7 python train.py --model_name RCAN_bic_deconv_50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 50 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4

# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_bic_clean --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 200  --mark RCAN_noise --stack_input True  --lr 2e-4

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_un_transpose_large_fft_50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 50 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_un_transpose_large_fft_100 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_un_transpose_small_fft_clean_fft --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 



# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_un_transpose_large_fft_25 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 25 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 

# CUDA_VISIBLE_DEVICES=7 python train.py --model_name RCAN_un_transpose_large_fft_clean --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_transpose_large_fft_clean/checkpoints/epoch_42_model.pth




# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_un_pixel_shuffle_large_fft_clean_fixed --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 

# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_pixel_shuffle_large_fft_clean_loss/checkpoints/epoch_20_model.pth

# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_un_transpose_large_fft_clean_sinc_50 --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_transpose_large_fft_clean_sinc_50/checkpoints/epoch_35_model.pth





# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_un_transpose_large_fft_clean_sinc_0 --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_transpose_large_fft_clean_sinc_0/checkpoints/epoch_16_model.pth

# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_un_transpose_large_fft_clean_icnr --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_transpose_large_fft_clean_sinc_50/checkpoints/epoch_16_model.pth

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_all_data --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max -1 --train_data_num -1  --mark RCAN_noise --stack_input True  --lr 2e-4 



# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_un_transpose_large_fft_clean_sinc_100 --sinc_prob 100 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_transpose_large_fft_clean_sinc_100/checkpoints/epoch_16_model.pth
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_un_transpose_large_fft_clean_fixed --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 


# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_un_pixel_shuffle_large_fft_clean_sinc_0 --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_pixel_shuffle_large_fft_clean_sinc_0/checkpoints/epoch_16_model.pth

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_un_pixel_shuffle_large_fft_clean --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_pixel_shuffle_large_fft_clean_sinc_0/checkpoints/epoch_16_model.pth



# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_un_pixel_shuffle_large_fft_clean_sinc_50 --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_pixel_shuffle_large_fft_clean_sinc_50/checkpoints/epoch_16_model.pth
# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 


# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_bn --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_transpose_large_fft_clean_fixed --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 

# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_un_transpose_large_fft_clean_75 --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 75 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 


# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_un_transpose_large_fft_75 --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_transpose_large_fft_75/checkpoints/epoch_48_model.pth


# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_un_transpose_large_fft_clean_sinc_100 --sinc_prob 100 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_transpose_large_fft_clean_sinc_100/checkpoints/epoch_16_model.pth





# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_un_transpose_large_fft_75 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 75 --train_data_num 2000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_transpose_large_fft_75/checkpoints/epoch_44_model.pth

# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_un_transpose_large_fft_clean_for --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 2000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise_ago/ALL/RCAN_un_transpose_large_fft_clean/checkpoints/epoch_60_model.pth




# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_un_pixel_shuffle_large_fft_clean_sinc_100 --sinc_prob 100 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_noise/ALL/RCAN_un_pixel_shuffle_large_fft_clean_sinc_100/checkpoints/epoch_16_model.pth


# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_0 --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 


# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_0 --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 25 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 


# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_25 --sinc_prob 25 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_50 --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 25 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_75 --sinc_prob 75 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 25 --train_data_num 200  --mark RCAN_noise --stack_input True  --lr 2e-4 


# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_0_rand --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 

# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_0_rand --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_var_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 



# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_0_1 --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 25 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 1


# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_100 --sinc_prob 100 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 25 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9

# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_un_transpose_large_fft_clean_sinc_100 --sinc_prob 100 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 25 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9

# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_un_transpose_large_fft_clean_sinc_50 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 25 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9



# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_bicubic --sinc_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 25 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9



# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_75 --sinc_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 25 --train_data_num 20000 --mark RCAN_noise --stack_input True  --lr 2e-4 --num_workers 0 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_4x4_4x_RCAN_noise/ALL/RCAN_pixel_shuffle_large_fft_clean_loss_sinc_75/checkpoints/epoch_08_model.pth




# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_75_wavelet --sinc_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 25 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9


# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_0_conv --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9


# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_75_conv --sinc_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9


# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/psf_less/

# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_ablation_supervised --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/psf_less/

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_unsupervised --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/psf_less/



# sinc50 for loss function ablation study
CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed_sinc50 --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/psf_less/

CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_ablation_supervised_sinc50 --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/psf_less/

CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_ablation_unsupervised_sinc50  --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/psf_less/

# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_flow_pixelshuffle_clean  --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/psf_less/





# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_unsupervised_transpose --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9


# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_50_larger_defocus --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9

# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_75_larger_defocus --sinc_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9

# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_ablation_mixed_larger_defocus --sinc_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9


# CUDA_VISIBLE_DEVICES=7 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_0_larger_defocus --sinc_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9



# -4, -5, -6
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_ablation_mixed_sinc50 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_1/

# # 三个psf -6, 6, 0
# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_ablation_mixed_sinc50_1 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_3/



# # 大散焦  10
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_mixed_sinc50_defocus_10 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_defocus_10/


# # 大散焦  6
# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_ablation_mixed_sinc50_defocus_6 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_noise --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_defocus_6/



# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_ablation_mixed_sinc50_0 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 0

# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_ablation_mixed_sinc50_3 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 3

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_mixed_sinc50_6 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 6

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed_sinc50_9 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 9

# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_ablation_mixed_sinc50_0-3+3 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 0,-3,3

# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_ablation_mixed_sinc50_0-3+3-6+6 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 0,-3,3,-6,6

# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_ablation_mixed_sinc50_0-3+3-6+6-9+9 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 0,-3,3,-6,6,-9,9

# CUDA_VISIBLE_DEVICES=7 python train.py --model_name RCAN_ablation_mixed_sinc50_all --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name ALL



# # sinc 0
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_ablation_mixed_sinc0_0 --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 0

# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_ablation_mixed_sinc0_3 --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 3

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_mixed_sinc0_6 --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 6

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed_sinc0_9 --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 9

# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_ablation_mixed_sinc0_0-3+3 --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 0,-3,3

# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_ablation_mixed_sinc0_0-3+3-6+6 --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 0,-3,3,-6,6

# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_ablation_mixed_sinc0_0-3+3-6+6-9+9 --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name 0,-3,3,-6,6,-9,9

# CUDA_VISIBLE_DEVICES=7 python train.py --model_name RCAN_ablation_mixed_sinc0_all --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf/ --psf_name ALL



# CUDA_VISIBLE_DEVICES=0 python train.py --model_name RCAN_ablation_mixed_sinc0_real_27 --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/ --psf_name 27 --train_data_num 2000 

# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_ablation_mixed_sinc50 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/psf_real/ --psf_name ALL

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed_sinc75 --sinc_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/psf_real/ --psf_name ALL

# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_pixel_shuffle_large_fft_clean_loss_sinc_75_conv --sinc_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/psf_real/ --psf_name ALL

# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_flow_pixelshuffle_clean --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/  --psf_name ALL



# 两个实验  
# sinc 函数 0， 50，

CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_ablation_mixed_sinc0_all --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/ --psf_name ALL --train_data_num 20000 

CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_mixed_sinc50_all --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/ --psf_name ALL --train_data_num 20000 


# 聚焦范围 

# 50
CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed_sinc50_50 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/ --psf_name 50 --train_data_num 20000 


# 27
CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_ablation_mixed_sinc50_27 --sinc_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/ --psf_name 27 --train_data_num 20000 


CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_flow_pixelshuffle_clean --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/  --psf_name ALL


CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_ablation_mixed_sinc75_all --sinc_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/  --psf_name ALL

CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_flow_pixelshuffle_clean2 --sinc_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/  --psf_name ALL


#  40, 45, 50, 55, 60 




# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_ablation_mixed_sinc25_all --mixed_prob 25 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/ --psf_name ALL --train_data_num 20000 

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_mixed_sinc50_all --mixed_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/ --psf_name ALL --train_data_num 20000 


# 聚焦范围 

CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed_sinc0_all --mixed_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_mixed/ --psf_name ALL --train_data_num -1 

# # 50
# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed_sinc50_50 --mixed_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/ --psf_name 50 --train_data_num 20000 

# # 27
CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_ablation_mixed_sinc25_all --mixed_prob 25 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_mixed/ --psf_name ALL --train_data_num -1 


CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_ablation_mixed_sinc50_all --mixed_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_mixed/ --psf_name ALL --train_data_num -1 


CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_ablation_mixed_sinc75_all --mixed_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_mixed/  --psf_name ALL --train_data_num -1 


# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_ablation_mixed_sinc75_all --mixed_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/  --psf_name ALL --train_data_num 20000 

# CUDA_VISIBLE_DEVICES=1 python train.py --model_name RCAN_flow_pixelshuffle_clean2 --mixed_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark RCAN_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_real/  --psf_name ALL

# 20230529  
# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed_sinc50_all --mixed_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf_simulation_defocus/ --psf_name ALL --train_data_num -1 


# 20230529  
# CUDA_VISIBLE_DEVICES=3 python train.py --model_name pix2pix --mixed_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf_mixed/ --psf_name ALL --train_data_num -1 


# CUDA_VISIBLE_DEVICES=6 python train_test_real.py --model_name RCAN_ablation_mixed_sinc75_all --mixed_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_single --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/psf_mixed/  --psf_name ALL --train_data_num -1 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_4x4_4x_RCAN_kernel/ALL/RCAN_ablation_mixed_sinc75_all/checkpoints/epoch_146_model.pth 


# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_mixed_sinc50_all --mixed_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_kernel_defocus --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf_simulation_defocus/ --psf_name ALL --train_data_num -1 


# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_mixed_sinc75_all --mixed_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 50  --mark RCAN_kernel --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf_real/  --psf_name 27 --train_data_num 200 --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_4x4_4x_RCAN_kernel/ALL/RCAN_ablation_mixed_sinc75_all/checkpoints/epoch_146_model.pth 


# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_ablation_mixed_sinc75_all_resize --mixed_prob 75 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf_mixed/  --psf_name ALL --train_data_num -1 


# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_ablation_mixed_sinc0_all_resize --mixed_prob 0 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf_mixed/  --psf_name ALL --train_data_num -1 


# CUDA_VISIBLE_DEVICES=7 python train.py --model_name RCAN_ablation_mixed_sinc50_all_resize --mixed_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf_mixed/  --psf_name ALL --train_data_num -1 



# 8*8
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_mixed_sinc0_all --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/  --psf_name ALL --train_data_num -1 

# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed_sinc75_all --mixed_prob 75 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/  --psf_name ALL --train_data_num -1 


# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_ablation_mixed_sinc75_all_resize --mixed_prob 75 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/  --psf_name ALL --train_data_num -1 --resize True

# CUDA_VISIBLE_DEVICES=7 python train.py --model_name RCAN_ablation_mixed_sinc50_all --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/  --psf_name ALL --train_data_num -1 --resize True

CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_flow_pixelshuffle_clean2 --mixed_prob 100 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/  --psf_name ALL --train_data_num -1 --resize True




# CUDA_VISIBLE_DEVICES=4 python train_test_real.py --model_name RCAN_un_pixel_shuffle --mixed_prob 100 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_single --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf_real/  --psf_name 27 --path_for_single /root/user/lfh/dataset/20230517-crop/4x4/27


# CUDA_VISIBLE_DEVICES=4 python train_test_real.py --model_name RCAN_ablation_mixed_sinc75_all_resize --mixed_prob 100 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_single --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf_real/  --psf_name 20 --path_for_single /root/user/lfh/dataset/20230517-crop/4x4/20 --repeat 1000



# CUDA_VISIBLE_DEVICES=4 python train_test_real.py --model_name hash_v1 --mixed_prob 100 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_single --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf_real/  --psf_name 20 --path_for_single /root/user/lfh/dataset/20230517-crop/4x4/20 --repeat 1000



# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RepSR --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/  --psf_name ALL --train_data_num -1 


# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RepSR2 --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/  --psf_name ALL --train_data_num -1 


# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RepSR3 --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark RCAN_resize --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/  --psf_name ALL --train_data_num -1 



# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed_sinc0_all --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark debug --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/ --psf_name ALL --train_data_num 20000


# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RepSR --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark debug --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/ --psf_name ALL --train_data_num 20000


# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RepSR_sinc75 --mixed_prob 75 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100  --mark debug --stack_input True  --lr 2e-4 --angnum 9  --psf_path /root/user/lfh/dataset/train_psf/psf/ --psf_name ALL --train_data_num 20000


# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_ablation_mixed --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark debug --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf/



# 测试新的psf 8*8的


# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_mixed --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real/


# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_ablation_mixed --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark defocus_10 --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_defocus/


# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RepSR --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark defocus_10 --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_defocus/

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_flow_pixelshuffle_clean2 --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark defocus_10 --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_defocus/


CUDA_VISIBLE_DEVICES=2 python train.py --model_name RepSR_bn --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real/



CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_ablation_mixed_sinc50_all --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real/


CUDA_VISIBLE_DEVICES=3 python train.py --model_name RepSR --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real/


# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RepSR_sinc50 --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real/


# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RepSR_wo_bn --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real/


# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RepSR_sinc75 --mixed_prob 75 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real/



# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RepSR_wo_res --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real/


# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RepSR_RCAN --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real/



# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_ablation_mixed_sinc50_all --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_5 --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_5/



# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RepSR --mixed_prob 0 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_5 --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_5/



# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real/


# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 100  --mark psf_real_f4 --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4/ --path_pre_pth /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_psf_real/ALL/RCAN_small/checkpoints/epoch_101_model.pth


# /root/user/lfh/experiments/UnsupervisedLFSR/log/SR_8x8_4x_psf_real_fundus/ALL/RCAN_small/checkpoints/epoch_27_model.pth


# CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/


# CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 4 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_wenhui --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_wenhui/


# CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 8 --scale_factor 8 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_wenhui --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_wenhui/


# CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/


# CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_small_bic --mixed_prob 50 --angRes 8 --scale_factor 8 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_wenhui --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_wenhui/


CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_small_SA --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/

CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_large --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/

CUDA_VISIBLE_DEVICES=6 python train.py --model_name pix2pix --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/

CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_simulation --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf/ --psf_name 0,1

CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_small8 --mixed_prob 50 --angRes 8 --scale_factor 8 --batch_size 16  --noise_std_max 100   --mark psf_wenhui --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_wenhui/ --test_data_name data_wenhui --train_data_num 200

CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_small6_2 --mixed_prob 50 --angRes 6 --scale_factor 6 --batch_size 16  --noise_std_max 100   --mark psf_wenhui --stack_input True  --lr 2e-4 --angnum 30 --psf_path /root/user/lfh/dataset/train_psf/psf_wenhui/ --test_data_name data_wenhui --train_data_num 200

CUDA_VISIBLE_DEVICES=5 python train.py --model_name RCAN_small6_2 --mixed_prob 50 --angRes 6 --scale_factor 6 --batch_size 16  --noise_std_max 100   --mark psf_wenhui_36view --stack_input True  --lr 2e-4 --angnum 30 --psf_path /root/user/lfh/dataset/train_psf/psf_wenhui/ --test_data_name data_wenhui --train_data_num 20000


CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf_8x8x4_scan --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/ 


CUDA_VISIBLE_DEVICES=2 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 8 --scale_factor 2 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf_8x8x2 --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/ 

CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf_8x8x4 --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/ 


CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_small8 --mixed_prob 50 --angRes 8 --scale_factor 8 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf_8x8x8 --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/ --test_data_name test_data_no_scan


CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_small8 --mixed_prob 50 --angRes 8 --scale_factor 8 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf_8x8x8_64view --stack_input True  --lr 2e-4 --angnum 40 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/ --test_data_name test_data_no_scan


CUDA_VISIBLE_DEVICES=4 python train.py --model_name RCAN_small8_5 --mixed_prob 50 --angRes 8 --scale_factor 8 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf_8x8x8_64view --stack_input True  --lr 2e-4 --angnum 40 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/ --test_data_name test_data_no_scan


CUDA_VISIBLE_DEVICES=6 python train.py --model_name RCAN_small8_6 --mixed_prob 50 --angRes 8 --scale_factor 8 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf_8x8x8_64view --stack_input True  --lr 2e-4 --angnum 40 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/ --test_data_name test_data_no_scan




CUDA_VISIBLE_DEVICES=3 python train.py --model_name RCAN_small --mixed_prob 50 --angRes 8 --scale_factor 4 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f6_new_all_psf_8x8x4_scan --stack_input True  --lr 2e-4 --angnum 9 --psf_path /root/user/lfh/dataset/train_psf/psf_f6_20230807/ 



CUDA_VISIBLE_DEVICES=5 python train.py --model_name RepSR_bn --mixed_prob 50 --angRes 8 --scale_factor 8 --batch_size 16  --noise_std_max 100 --train_data_num 20000  --mark psf_real_f4_new_all_psf_8x8x8_64view --stack_input True  --lr 2e-4 --angnum 40 --psf_path /root/user/lfh/dataset/train_psf/psf_real_f4_new/ --test_data_name test_data_no_scan
