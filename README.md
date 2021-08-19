[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://github.com/Ha0Tang/AttentionGAN/blob/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/Ha0Tang/AttentionGAN)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/Ha0Tang/AttentionGAN/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)

# Self-Supervised Attention-Guided GAN for Unpaired Image-to-Image Translation

AttentionGAN: Unpaired Image-to-Image Translation using Attention-Guided Generative Adversarial Networks.<br>
[Hao Tang](http://disi.unitn.it/~hao.tang/)<sup>1</sup>, [Hong Liu](https://scholar.google.com/citations?user=4CQKG8oAAAAJ&hl=en)<sup>2</sup>, [Dan Xu](http://www.robots.ox.ac.uk/~danxu/)<sup>3</sup>, [Philip H.S. Torr](https://scholar.google.com/citations?user=kPxa2w0AAAAJ&hl=en)<sup>3</sup> and [Nicu Sebe](http://disi.unitn.it/~sebe/)<sup>1</sup>. <br> 
<sup>1</sup>University of Trento, Italy, <sup>2</sup>Peking University, China, <sup>3</sup>University of Oxford, UK.<br>
The repository offers the official implementation of our paper in PyTorch.

## AttentionGAN Training/Testing
- Download a dataset using the previous script (e.g., horse2zebra).
- To view training results and loss plots, run `python -m visdom.server` and click the URL [http://localhost:8097](http://localhost:8097).
- Train a model:
```
bash ./scripts/train_attentiongan.sh
```
- To see more intermediate results, check out `./checkpoints/horse2zebra_attentiongan/web/index.html`.
- How to continue train? Append `--continue_train --epoch_count xxx` on the command line.
- Test the model:
```
bash ./scripts/test_attentiongan.sh
```
- The test results will be saved to a html file here: `./results/horse2zebra_attentiongan/latest_test/index.html`.

## Generating Images Using Pretrained Model
- You need download a pretrained model (e.g., horse2zebra) with the following script:
```
bash ./scripts/download_attentiongan_model.sh horse2zebra
```
- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. 
- Then generate the result using
```
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_pretrained --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 5000 --epoch latest --saveDisk
```
The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory. Note that if you want to save the intermediate results and have enough disk space, remove `--saveDisk` on the command line.

- For your own experiments, you might want to specify --netG, --norm, --no_dropout to match the generator architecture of the trained model.

### Image Translation with Geometric Changes Between Source and Target Domains
For instance, if you want to run experiments of Selfie to Anime Translation. Usage: replace `attention_gan_model.py` and `networks` with the ones in the `AttentionGAN-geo` folder.

### Test the Pretrained Model 
Download data and pretrained model according above instructions.

`python test.py --dataroot ./datasets/selfie2anime/ --name selfie2anime_pretrained --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 5000 --epoch latest`

### Train a New Model
`python train.py --dataroot ./datasets/selfie2anime/ --name selfie2anime_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 286 --crop_size 256 --batch_size 4 --niter 100 --niter_decay 100 --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100`

### Test the Trained Model
`python test.py --dataroot ./datasets/selfie2anime/ --name selfie2anime_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 5000 --epoch latest`

## Evaluation Code
- [FID](https://github.com/bioinf-jku/TTUR): Official Implementation
  Install Steps: `conda create -n python36 pyhton=3.6 anaconda` and `pip install --ignore-installed --upgrade tensorflow==1.13.1`. If you encounter the issue `AttributeError: module 'scipy.misc' has no attribute 'imread'`, please do `pip install scipy==1.1.0`.

## Acknowledgments
This source code is inspired by [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

