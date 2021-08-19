"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time, datetime
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from statistics import mean

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    ax = {}
    fig = {}

    ax_rotation = {}
    fig_rotation = {}

    D_A_losses = []
    D_B_losses = []
    G_A_losses = []
    G_B_losses = []
    cycle_A_losses = []
    cycle_B_losses = []

    idt_A_losses = []
    idt_B_losses = []

    D_A_epoch_losses = []
    D_B_epoch_losses = []
    G_A_epoch_losses = []
    G_B_epoch_losses = []
    cycle_A_epoch_losses = []
    cycle_B_epoch_losses = []

    idt_A_epoch_losses = []
    idt_B_epoch_losses = []

    d_class_losses_A = []
    d_class_losses_B = []
    g_class_losses_A = []
    g_class_losses_B = []
    
    d_class_epoch_losses_A = []
    d_class_epoch_losses_B = []
    g_class_epoch_losses_A = []
    g_class_epoch_losses_B = []

    save_dir = 'results/results_losses/' + str(datetime.datetime.now()).replace(' ', '@')
    os.makedirs(save_dir)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        for _, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            ####### Self-Supervised Task #######
            batch_len = opt.batch_size
            model.transfer_batch_size(batch_len)
            ####### Self-Supervised Task #######
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            losses = model.get_current_losses()
            rotation_losses = model.transfer_rotation_loss()
            for key, value in losses.items():
                if key == 'D_A':
                    D_A_losses.append(value)
                elif key == 'G_A':
                    G_A_losses.append(value)
                elif key == 'cycle_A':
                    cycle_A_losses.append(value)
                elif key == 'D_B':
                    D_B_losses.append(value)
                elif key == 'G_B':
                    G_B_losses.append(value)
                elif key == 'cycle_B':
                    cycle_B_losses.append(value)
                elif key == 'idt_A':
                    idt_A_losses.append(value)
                elif key == 'idt_B':
                    idt_B_losses.append(value)

            d_class_losses_A.append(rotation_losses[0].item())
            d_class_losses_B.append(rotation_losses[1].item())
            g_class_losses_A.append(rotation_losses[2].item())
            g_class_losses_B.append(rotation_losses[3].item())

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                #losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        #################################  ITERATION  #########################################
        fig[epoch] = plt.figure()
        ax[epoch] = fig[epoch].add_subplot(1, 1, 1)

        ax[epoch].plot(range(len(D_A_losses)), D_A_losses, label="D_A loss")
        ax[epoch].plot(range(len(D_B_losses)), D_B_losses, label="D_B loss")
        ax[epoch].plot(range(len(G_A_losses)), G_A_losses, label="G_A loss")
        ax[epoch].plot(range(len(G_B_losses)), G_B_losses, label="G_B loss")
        ax[epoch].plot(range(len(cycle_A_losses)), cycle_A_losses, label="cycle_A loss")
        ax[epoch].plot(range(len(cycle_B_losses)), cycle_B_losses, label="cycle_B loss")

        if opt.lambda_identity > 0.0: 
            ax[epoch].plot(range(len(idt_A_losses)), idt_A_losses, label="idt_A loss")
            ax[epoch].plot(range(len(idt_B_losses)), idt_B_losses, label="idt_B loss")
            
        ax[epoch].set_xlabel('iter')
        ax[epoch].set_ylabel('loss')

        ax[epoch].legend()
        ax[epoch].grid()

        plt.savefig(save_dir + '/iter_losses_' + str(epoch) + '.png') 
        
        """Rotation Loss"""
        fig_rotation[epoch] = plt.figure()
        ax_rotation[epoch] = fig_rotation[epoch].add_subplot(1, 1, 1)

        ax_rotation[epoch].plot(range(len(d_class_losses_A)), d_class_losses_A, label="D_rotation A loss")
        ax_rotation[epoch].plot(range(len(d_class_losses_B)), d_class_losses_B, label="D_rotation B loss")
        ax_rotation[epoch].plot(range(len(g_class_losses_A)), g_class_losses_A, label="G_rotation A loss")
        ax_rotation[epoch].plot(range(len(g_class_losses_B)), g_class_losses_B, label="G_rotation B loss")

        ax_rotation[epoch].set_xlabel('iter')
        ax_rotation[epoch].set_ylabel('loss')

        ax_rotation[epoch].legend()
        ax_rotation[epoch].grid()

        plt.savefig(save_dir + '/iter_rotation_losses_' + str(epoch) + '.png') 
        #################################  ITERATION  #########################################

        ###################################  EPOCH  ###########################################
        D_A_epoch_losses.append(mean(D_A_losses))
        D_B_epoch_losses.append(mean(D_B_losses))
        G_A_epoch_losses.append(mean(G_A_losses))
        G_B_epoch_losses.append(mean(G_B_losses))
        cycle_A_epoch_losses.append(mean(cycle_A_losses))
        cycle_B_epoch_losses.append(mean(cycle_B_losses))

        if opt.lambda_identity > 0.0: 
            idt_A_epoch_losses.append(mean(idt_A_losses))
            idt_B_epoch_losses.append(mean(idt_B_losses))

        """Rotation Loss"""
        d_class_epoch_losses_A.append(mean(d_class_losses_A))
        d_class_epoch_losses_B.append(mean(d_class_losses_B))
        g_class_epoch_losses_A.append(mean(g_class_losses_A))
        g_class_epoch_losses_B.append(mean(g_class_losses_B))
        ###################################  EPOCH  ###########################################

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    
    #################################### EPOCH ########################################
    fig_e = plt.figure()
    ax_e = fig_e.add_subplot(1, 1, 1)

    ax_e.plot(range(len(D_A_epoch_losses)), D_A_epoch_losses, label="D_A loss")
    ax_e.plot(range(len(D_B_epoch_losses)), D_B_epoch_losses, label="D_B loss")
    ax_e.plot(range(len(G_A_epoch_losses)), G_A_epoch_losses, label="G_A loss")
    ax_e.plot(range(len(G_B_epoch_losses)), G_B_epoch_losses, label="G_B loss")
    ax_e.plot(range(len(cycle_A_epoch_losses)), cycle_A_epoch_losses, label="cycle_A loss")
    ax_e.plot(range(len(cycle_B_epoch_losses)), cycle_B_epoch_losses, label="cycle_B loss")

    if opt.lambda_identity > 0.0: 
        ax_e.plot(range(len(idt_A_epoch_losses)), idt_A_epoch_losses, label="idt_A loss")
        ax_e.plot(range(len(idt_B_epoch_losses)), idt_B_epoch_losses, label="idt_B loss")
    
    ax_e.set_xlabel('epoch')
    ax_e.set_ylabel('loss')
    
    ax_e.legend()
    ax_e.grid()

    plt.savefig(save_dir + '/result_epoch_losses.png')
     
    """Rotation Loss"""
    fig_e_rotation = plt.figure()
    ax_e_rotation = fig_e_rotation.add_subplot(1, 1, 1)

    ax_e_rotation.plot(range(len(d_class_epoch_losses_A)), d_class_epoch_losses_A, label="D_rotation A loss")
    ax_e_rotation.plot(range(len(d_class_epoch_losses_B)), d_class_epoch_losses_B, label="D_rotation B loss")
    ax_e_rotation.plot(range(len(g_class_epoch_losses_A)), g_class_epoch_losses_A, label="G_rotation A loss")
    ax_e_rotation.plot(range(len(g_class_epoch_losses_B)), g_class_epoch_losses_B, label="G_rotation B loss")

    ax_e_rotation.set_xlabel('epoch')
    ax_e_rotation.set_ylabel('loss')

    ax_e_rotation.legend()
    ax_e_rotation.grid()

    plt.savefig(save_dir + '/result_rotation_epoch_losses.png') 
    #################################### EPOCH ########################################