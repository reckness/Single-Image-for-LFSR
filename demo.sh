
python train.py --model_name RCAN_bilinear_flow --angRes 8 --scale_factor 4 --batch_size 8  --noise_var_max 0 --train_data_num 5120 --channels 64 --n_group 8 --n_block 1 --mark Hash_new_test --stack_input True  --lr 5e-4 --path_pre_pth D:/workspace/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_5120/ALL/RCAN_bilinear_flow/checkpoints/RCAN_bilinear_flow_8x8_4x_epoch_94_model.pth --mark RCAN_bilinear_flow


python train.py --model_name RCAN_bilinear_flow --angRes 8 --scale_factor 4 --batch_size 8  --noise_var_max 0 --train_data_num 5120 --channels 64 --n_group 8 --n_block 1 --mark Hash_new_test --stack_input True  --lr 5e-4 --path_pre_pth D:/workspace/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_5120/ALL/RCAN_bilinear_flow/checkpoints/RCAN_bilinear_flow_8x8_4x_epoch_94_model.pth --mark RCAN_bilinear_flow




python inference_stack.py --model_name RCAN_transpose --angRes 8 --scale_factor 4 --path_pre_pth 'D:\workspace\experiments\UnsupervisedLFSR\log\SR_8x8_4x_RCAN_5120\ALL\RCAN_transpose\checkpoints/RCAN_transpose_8x8_4x_epoch_49_model.pth --mark RCAN_transpose


python train_test.py --model_name RepLFSR --angRes 8 --scale_factor 4 --batch_size 1  --noise_var_max 0 --train_data_num 1 --channels 64 --n_group 1 --n_block 8 --angnum -1 --mark 20230302_fft_conv_single_view64_res_refocus


python train.py --model_name RepLFSR --angRes 8 --scale_factor 4 --batch_size 1  --noise_var_max 0 --train_data_num 1 --channels 64 --n_group 1 --n_block 8 --angnum -1 --mark 20230302_fft_single_view64_res_refocus_pretrain





# pix2pix
python train_test.py --model_name pix2pix --angRes 8 --scale_factor 4 --batch_size 8  --noise_var_max -1 --train_data_num 128 --channels 64 --n_group 1 --n_block 8 --angnum -1 --mark 20230306_fft




####################

python inference_stack.py --model_name RepLFSR --angRes 4 --scale_factor 2 --path_pre_pth ../experiments/BasicLFSR/log/SR_4x4_2x_Rep/ALL/RepLFSR/checkpoints/RepLFSR_4x4_2x_epoch_30_model.pth --mark RepLFSR
python inference_stack.py --model_name RCAN_bilinear_flow --angRes 8 --scale_factor 4 --path_pre_pth D:/workspace/experiments/UnsupervisedLFSR/log/SR_8x8_4x_RCAN_5120/ALL/RCAN_bilinear_flow/checkpoints/RCAN_bilinear_flow_8x8_4x_epoch_94_model.pth --mark RCAN_bilinear_flow
python inference_stack.py --model_name swinir --angRes 8 --scale_factor 4 --path_pre_pth D:\workspace\experiments\UnsupervisedLFSR\log\SR_8x8_4x_SWIN_transformer_5120\ALL\swinir_bilinear\checkpoints/SWIN_transformer_5120_8x8_4x_epoch_94_model.pth --mark SWIN_transformer_5120
