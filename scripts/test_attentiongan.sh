set -ex
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 5
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 10
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 15
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 20
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 25
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 30
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 35
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 40
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 45
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 50
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 55
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 
###
python test.py --dataroot ./datasets/apple2orange --name apple2orange_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk --weight_rotation_loss_g 0.1 --load_iter 60
###
python -m pytorch_fid ./datasets/apple2orange/testB ./results/apple2orange_attentiongan/Rot_d=0.5/test_latest/images/fakeB 