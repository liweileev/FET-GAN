import os
import torch
from natsort import natsorted
import random
from models.FET_model import FET_model as FETModel
from utils.util import get_option, save_images, print_save_options, make_test_data
from utils import html
import random

opt = get_option('./configs/font_effects.yaml')
opt['K'] = 8
opt['name'] = 'TextEffects' # model's name (folder name): 'TextEffects' / 'Fonts100'
opt['testresults_dir'] = opt['testresults_dir'] + '/TextEffects' # '/TextEffects' / '/Fonts100'
opt['isTrain'] = False # train(True) or test(False)
opt['batch_size'] = 1
opt['num_threads'] = 0
opt['display_id'] = -1
opt['load_size'] = 160
opt['crop_size'] = 128
opt['load_iter'] = 30

opt['testsource'] = None #'./testimgs/TextEffects/FontEffects_sources/github.png' #None # single source image for test
opt['testsource_dir'] = './testimgs/TextEffects/TextEffects_sources/' # source image folder for test './testimgs/Fonts100/Fonts100_sources/', './testimgs/TextEffects/TextEffects_sources/'

opt['testrefs'] = None #'./testimgs/TextEffects/FontEffects_refs/derby' #None # single ref images folder for test
opt['testrefs_dir'] = './testimgs/TextEffects/TextEffects_refs' # multiple reference images folders for test './testimgs/Fonts100/Fonts100_refs', './testimgs/TextEffects/TextEffects_refs'

print_save_options(opt)

assert (opt['testsource'] or opt['testsource_dir'])
assert (opt['testrefs'] or opt['testrefs_dir'])

if opt['testsource']:
    source_paths = [opt['testsource']]
else:
    source_paths = [os.path.join(opt['testsource_dir'], f) for f in os.listdir(opt['testsource_dir'])]
    
if opt['testrefs']:
    finetune_paths = [os.path.join(opt['testrefs'], f) for f in os.listdir(opt['testrefs'])]
    finetune_size = len(finetune_paths)
    if opt['K'] <= finetune_size:
            ref_paths_ = random.sample(finetune_paths, opt['K'])
    else:
        ref_paths_ += finetune_paths
        cnt = finetune_size
        while opt['K'] > cnt:
            if opt['K'] - cnt > finetune_size:
                ref_paths_ += finetune_paths
                cnt += finetune_size
            else:
                ref_paths_ += random.sample(finetune_paths, opt['K'] - cnt)
                cnt += opt['K'] - cnt
    ref_paths = [ref_paths_]
else:
    ref_paths = []
    for dir in natsorted(os.listdir(opt['testrefs_dir'])):
        current_paths = [os.path.join(opt['testrefs_dir'], dir, f) for f in os.listdir(os.path.join(opt['testrefs_dir'],dir))]
        ref_paths.append(current_paths)

model = FETModel(opt)      # create a model given other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()

# create a website
web_dir = os.path.join(opt['testresults_dir'], opt['name'], 'test_%s' % (opt['epoch']))  # define the website directory
webpage = html.HTML(web_dir, 'Test Experiment = %s, Epoch = %s' % (opt['name'], opt['epoch']))

cnt = 1
for source_path in source_paths:
    for _,refs_path in enumerate(ref_paths):
        source = make_test_data(source_path, opt['load_size'], opt['crop_size'])  # 1*3*128*128
        refs = torch.zeros(opt['K'], opt['input_nc'], opt['crop_size'], opt['crop_size'])  # K*3*128*128
        for i,ref_path in enumerate(refs_path):
            refs[i] = make_test_data(ref_path, opt['load_size'], opt['crop_size'])
        refs.unsqueeze_(0) # 1*K*3*128*128
        data = {'source':source, 'refs':refs}
        
        model.set_input(data)
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        print('processing: %d/%d' % (cnt, len(source_paths) * len(ref_paths)))
        save_images(webpage, cnt, visuals, opt['K'])
        cnt += 1

webpage.save()  # save the HTML