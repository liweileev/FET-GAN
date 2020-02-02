"""This module contains simple helper functions """
import os
import torch
from natsort import natsorted
import numpy as np
from PIL import Image
import ntpath
import yaml
import torch.utils.data as data
from torchvision import transforms
import random

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        if len(input_image.size()) == 4:    
            image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        elif len(input_image.size()) == 5:   # refs
            image_numpy = image_tensor[0][0].cpu().float().numpy()
            for i in range(1,len(image_tensor[0])):
                image_numpy = np.append(image_numpy, image_tensor[0][i].cpu().float().numpy(), axis=2)  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling [-1,1]->[0,255]
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_images(webpage, id, visuals, K, width=128):
    """Save images to the disk.
    This function will also save images stored in 'visuals' to the HTML file specified by 'webpage'.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        id (int)                -- test id for name the output pairs
        width (int)              -- the images will be resized to width x width
    """
    image_dir = webpage.get_image_dir()
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s_%s.png' % (id, label)
        save_path = os.path.join(image_dir, image_name)
        save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, K, width=width)

def make_test_data(imgPath, size, crop):
    transform_list = []
    if size != None:
        transform_list.append(transforms.Resize(size))
    if crop != None:
        transform_list.append(transforms.CenterCrop(crop))
    transform_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    return transform(Image.open(imgPath).convert('RGB')).unsqueeze_(0)

def make_finetune_data(imgDir, size, crop, K, batchsize):

    # transform
    transform_list = []
    osize = [size, size]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    transform_list.append(transforms.CenterCrop(crop))  # CenterCrop
    # transform_list.append(transforms.RandomCrop(crop))    # RandomCrop
    # transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_finetune = transforms.Compose(transform_list)

    source_bs = torch.zeros([batchsize, 3, crop, crop])
    source_label_bs = torch.zeros([batchsize])
    target_bs = torch.zeros([batchsize, 3, crop, crop])
    refs_bs = torch.zeros([batchsize, K, 3, crop, crop])
    refs_label_bs = torch.zeros([batchsize])

    finetune_paths = [os.path.join(imgDir, f) for f in os.listdir(imgDir)]
    finetune_size = len(finetune_paths)

    for i in range(batchsize):
        source_path = random.sample(finetune_paths, 1)
        source_img = Image.open(source_path[0]).convert('RGB')
        source = transform_finetune(source_img).unsqueeze_(0) # 3*256*256 

        ref_paths = []
        if K <= finetune_size:
            ref_paths = random.sample(finetune_paths, K)
        else:
            ref_paths += finetune_paths
            cnt = finetune_size
            while K > cnt:
                if K - cnt > finetune_size:
                    ref_paths += finetune_paths
                    cnt += finetune_size
                else:
                    ref_paths += random.sample(finetune_paths, K - cnt)
                    cnt += K - cnt
        refs = torch.zeros(K, 3, crop, crop)  # 3*256*256
        for j,ref_path in enumerate(ref_paths):
            ref_img = Image.open(ref_path).convert('RGB')
            ref = transform_finetune(ref_img).unsqueeze_(0)
            refs[j] = ref
        refs = refs.unsqueeze_(0)
        source_label = torch.tensor([-1])
        refs_label = torch.tensor([-1])
        target = source

        source_bs[i] = source
        source_label_bs[i] = source_label
        target_bs[i] = target
        refs_bs[i] = refs
        refs_label_bs[i] = refs_label

    return {'source': source_bs, 'source_label': source_label_bs, 'target':target_bs, 'refs': refs_bs, 'refs_label': refs_label_bs}


def get_option(path):
    """ Load options from yaml file.

    Parameters:
        path (str)  -- the path of yaml file
    """
    with open(path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)

def print_save_options(opt, isSaveOptions=True, SaveSuffix=''):
    """print and save options .

    Parameters:
        opt (NameSpace) -- a namespace contains all options
        isSaveOptions (bool) -- whether to save to the disk
        SaveSuffix -- during test, add suffix to save path
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(opt.items()):
        comment = ''
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

    if opt['isTrain']:
        expr_dir = os.path.join(opt['outputs_dir'], opt['name'])
        checkpoints_dir = os.path.join(opt['outputs_dir'], opt['name'], 'checkpoints')
        mkdirs([expr_dir, checkpoints_dir])
    else:
        if SaveSuffix != '':
            expr_dir = os.path.join(opt['testresults_dir'], opt['name']+ '_' + SaveSuffix)
        else:
            expr_dir = os.path.join(opt['testresults_dir'], opt['name'])
        mkdir(expr_dir)
    
    # save to the disk
    if isSaveOptions:
        file_name = os.path.join(expr_dir, 'opt.txt') 
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def imshow_jupyter(imgtensor):
    tensor_numpy = imgtensor.cpu().float().numpy()  # convert it into a numpy array
    tensor_numpy = (np.transpose(tensor_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling [-1,1]->[0,255]
    tensor_numpy = tensor_numpy.astype(np.uint8)
    import cv2
    import IPython
    _,ret = cv2.imencode('.jpg', tensor_numpy) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)
    