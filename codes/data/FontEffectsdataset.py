import os
import numpy as np
import torch
from natsort import natsorted
import glob
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.util import is_image_file, make_dataset
from PIL import Image
import random

class FontEffectsDataset(data.Dataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (option dicts) -- stores all the experiment options;
        """
        self.opt = opt
        self.K = opt['K']    
        
        self.fonteffects_dir = opt['fonteffects_dir']
        self.num_cls = opt['num_cls']                     
        self.clsdict = {}
        for i,source_cls in enumerate(source_cls for source_cls in natsorted(os.listdir(self.fonteffects_dir)) if not source_cls.startswith('.')):
            self.clsdict[str(source_cls)] = i
            if i >= self.num_cls-1:
                break

        self.transform_fonteffects = get_transform(self.opt)
        
    
    def __getitem__(self, index):
        """Return a source image and K effect ref images.

        Parameters:
            index (int)  -- a random integer for data indexing

        Returns a dictionary that contains source, refs
            source (tensor)       -- an image in the source domain
            refs (tensor)       -- K corresponding ref images in the target domain
        """
        idx = index
    
        for i, source_cls in enumerate(source_cls for source_cls in natsorted(os.listdir(self.fonteffects_dir))[:self.num_cls] if not source_cls.startswith('.')):
            for source_name in (source_name for source_name in natsorted(os.listdir(os.path.join(self.fonteffects_dir, source_cls))) if not source_name.startswith('.')):
                if is_image_file(source_name):
                    if idx != 0:
                        idx -= 1
                    else:
                        break
            if idx == 0 or i >= self.num_cls-1:
                break

        source_path = os.path.join(self.fonteffects_dir, source_cls, source_name)
        source_label = self.clsdict[source_cls] # B

        other_cls = [cls for cls in natsorted(os.listdir(self.fonteffects_dir))[:self.num_cls] if cls != source_cls and not cls.startswith('.')]
        ref_cls = random.choice(other_cls)
        ref_paths = random.sample(glob.glob(os.path.join(self.fonteffects_dir, ref_cls, "*")), self.K)
        refs_label = self.clsdict[ref_cls] # B

        target_path = os.path.join(self.fonteffects_dir, ref_cls, source_name)
        target_img = Image.open(target_path).convert('RGB')
        target = self.transform_fonteffects(target_img) # B*3*256*256

        source_img = Image.open(source_path).convert('RGB')
        source = self.transform_fonteffects(source_img) # B*3*256*256  

        refs = torch.zeros(self.K, self.opt['input_nc'], self.opt['crop_size'], self.opt['crop_size'])  # B*K*3*256*256
        
        for i,ref_path in enumerate(ref_paths):
            ref_img = Image.open(ref_path).convert('RGB')
            ref = self.transform_fonteffects(ref_img)
            refs[i] = ref
            
        return {'source': source, 'source_label': source_label, 'target':target, 'refs': refs, 'refs_label': refs_label}
    
    def __len__(self):
        """Return the total number of images in the dataset.
        """
        len = 0
        for i, source_cls in enumerate(source_cls for source_cls in natsorted(os.listdir(self.fonteffects_dir)) if not source_cls.startswith('.')):
            for source_name in (source_name for source_name in natsorted(os.listdir(os.path.join(self.fonteffects_dir, source_cls))) if not source_name.startswith('.')):
                if is_image_file(source_name):
                    len += 1
            if i >= self.num_cls-1:
                break

        return len
    
def get_transform(opt, method=Image.BICUBIC, convert=True):
    transform_list = []
    osize = [opt['load_size'], opt['load_size']]
    transform_list.append(transforms.Resize(osize, method))
    transform_list.append(transforms.CenterCrop(opt['crop_size']))  # CenterCrop

    if convert:
        transform_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        
    return transforms.Compose(transform_list)