import datetime
import time
from data import create_dataset
from models.FET_model import FET_model as FETModel
from utils.util import get_option, print_save_options
from utils.visualizer import Visualizer

opt = get_option('./configs/font_effects.yaml')
opt['name'] = 'font_effect_transfer' 
opt['isTrain'] = True # train(True) or test(False)
opt['name'] = 'FET_' + datetime.date.today().strftime("%Y%m%d") #'name of the experiment. It decides where to store samples and models'
opt['fonteffects_dir'] = './datasets/TextEffects/train/'
opt['K'] = 4
opt['gpu_ids'] = [4]
opt['load_size'] = 160
opt['crop_size'] = 128

print_save_options(opt)

dataset = create_dataset(opt)  # create a dataset given options in config

model = FETModel(opt)      # create a model given other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

total_iters = 0                # the total number of training iterations

train_start_time = time.time()
epoch_end_time = [train_start_time]
for epoch in range(opt['epoch_count'], opt['nepoch'] + opt['nepoch_decay'] + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch        
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt['print_freq'] == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt['batch_size']
            epoch_iter += opt['batch_size']
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt['display_freq'] == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt['update_html_freq'] == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), total_iters, len(dataset), time.time()-train_start_time, save_result)

            if total_iters % opt['print_freq'] == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt['batch_size']
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt['display_id'] > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), losses)
                    visualizer.display_train_time(time.time(), epoch_end_time)

            if total_iters % opt['save_latest_freq'] == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt['save_by_iter'] else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt['save_epoch_freq'] == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            
        epoch_end_time.append(time.time())
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt['nepoch'] + opt['nepoch_decay'], epoch_end_time[-1] - epoch_end_time[-2]))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.