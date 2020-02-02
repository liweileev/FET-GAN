import numpy as np
import os
import sys
import ntpath
import time
from utils.util import mkdirs, tensor2im, save_image
from . import html
from subprocess import Popen, PIPE
from scipy.misc import imresize

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags;
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt['display_id']
        self.use_html = opt['isTrain'] and not opt['no_html']
        self.win_size = opt['display_winsize']
        self.name = opt['name']
        self.port = opt['display_port']
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.vis = visdom.Visdom(server=opt['display_server'], port=opt['display_port'], env=opt['display_env'])
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <outputs_dir>/web/; images will be saved under <outputs_dir>/web/images/
            self.web_dir = os.path.join(opt['outputs_dir'], opt['name'], 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt['outputs_dir'], opt['name'], 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
           
    def display_current_results(self, visuals, total_iter, dataset_size, training_time, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) -- dictionary of images to display or save
            total_iter (int) -- the current iteration
            dataset_size (int) -- the number of iterations in one epoch
            training_time (str) -- the training time
            save_result (bool) -- if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = len(visuals)
            labels = ''
            images = []
            idx = 0
            for label, image in visuals.items():
                image_numpy = tensor2im(image)
                if image_numpy.shape[0] != image_numpy.shape[1]:    #for K refs
                    for i in range(int(image_numpy.shape[1]/image_numpy.shape[0])):
                        images.append(image_numpy[:,i*image_numpy.shape[0]:(i+1)*image_numpy.shape[0],:].transpose([2, 0, 1]))
                    ncols = len(visuals) + int(image_numpy.shape[1]/image_numpy.shape[0]) - 1
                    idx += int(image_numpy.shape[1]/image_numpy.shape[0])
                else:
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                labels += '%s\t' % label
                
            white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
            while idx % ncols != 0:
                images.append(white_image)
                idx += 1
            try:
                self.vis.images(images, nrow=ncols if ncols<8 else 8, win=self.display_id + 1, padding=2, opts=dict(title='images', caption=labels))
            except VisdomExceptionBase:
                self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = tensor2im(image)
                img_path = os.path.join(self.img_dir, '%d_%s.png' % (total_iter, label))
                save_image(image_numpy, img_path)

            # update website
            if dataset_size * (self.opt['nepoch'] + self.opt['nepoch_decay']) - total_iter >= self.opt['display_freq']:
                webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=60)
                webpage.add_pagetitle('Experiment: %s' % self.name)
                webpage.add_caption('Dataset: %s (%d samples)' % (self.opt['datasetname'], dataset_size))
            else:
                webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
                webpage.add_pagetitle('Experiment: %s' % self.name)
                webpage.add_caption('Dataset size: %d' % dataset_size)
                m, s = divmod(training_time, 60)
                h, m = divmod(m, 60)
                webpage.add_caption('Training time: %d:%02d:%02d' % (h, m, s))
            for i in range(total_iter, 0, -self.opt['display_freq']):
                webpage.add_header('iter %d: epoch [%d]' % (i, i//dataset_size + 1))
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = tensor2im(image)
                    img_path = '%d_%s.png' % (i, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, K=self.opt['K'], width=self.win_size)
            webpage.save()

    def display_train_time(self, now_time, epoch_end_time):
        """Display time information of training
        
        Parameters:
            train_start_time (str) -- The timestamp of training start
            epoch_end_time (list) -- The string list of timestamp constainting the end time of each epoch
        """
        message = '<p>Training: <span style="color:red">%d</span> epochs</p>' % (self.opt['nepoch'] + self.opt['nepoch_decay'])
        message += '<p>----------</p>'
        message += '<p>Start: %s' % time.strftime("%Y-%m-%d %H:%M:%S</p>", time.localtime(epoch_end_time[0]))
        seconds = now_time - epoch_end_time[0]
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        message += '<p>Spend: <span style="color:red">%d:%02d:%02d</span></p>' % (h, m, s)
        if len(epoch_end_time) > 1:            
            for i in range(len(epoch_end_time)-1):
                seconds = epoch_end_time[i+1] - epoch_end_time[i]
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                message += '<p>----------</p>'
                message += '<p>Epoch %s spend: %d:%02d:%02d</p>' % (i+1, h, m, s)
        try:            
            self.vis.text(message, win=self.display_id + 2, opts=dict(title='time info'))
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio - 1)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.opt['name'] + ' ---- loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message