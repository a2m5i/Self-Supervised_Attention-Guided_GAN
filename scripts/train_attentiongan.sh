set -ex
#train
python train.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 286 --crop_size 256 --batch_size 4 --niter 60 --niter_decay 0 --gpu_ids 1 --display_id 0 --display_freq 10 --print_freq 10 --netG out --weight_rotation_loss_g 0.0 
#train
python train.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 286 --crop_size 256 --batch_size 4 --niter 60 --niter_decay 0 --gpu_ids 1 --display_id 0 --display_freq 10 --print_freq 10 --netG out --weight_rotation_loss_g 0.1 
#train
python train.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 286 --crop_size 256 --batch_size 4 --niter 60 --niter_decay 0 --gpu_ids 1 --display_id 0 --display_freq 10 --print_freq 10 --netG out --weight_rotation_loss_g 0.2 
#train
python train.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 286 --crop_size 256 --batch_size 4 --niter 60 --niter_decay 0 --gpu_ids 1 --display_id 0 --display_freq 10 --print_freq 10 --netG out --weight_rotation_loss_g 0.3 
#train
python train.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 286 --crop_size 256 --batch_size 4 --niter 60 --niter_decay 0 --gpu_ids 1 --display_id 0 --display_freq 10 --print_freq 10 --netG out --weight_rotation_loss_g 0.4 
