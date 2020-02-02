import cv2
import os
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as bwdist

def PreProcess(img_path):
    img = cv2.imread(img_path)
    I = img[:,:,2]
    I2 = bwdist(I <= 100);
    I3 = bwdist(I > 100);

    img[:,:,0] = np.clip(I2,0,255);
    img[:,:,1] = np.clip(I3,0,255);

    return img

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

fontname_list = ['BaiQing', 'CaiDie', 'ChenPinPo', 'DaiYuTi', 'DingHei', 'FanSong', 'HuoChaiTi', 'JinHuaTi', 'LeZhenTi', 'LingYi', 'MengYuan', 'MoYuTi', 'NianZhen', 'PuHui', 'PuYueTi', 'SuBai', 'YiFeng', 'YunSongTi', 'YuYi', 'ZhaoPaiTi']
for idx, fontname in enumerate(fontname_list):
    print('%d: %s' % (idx, fontname))
    for i in range(1,838):
        inputpath = '../../ChineseFonts/' + fontname + '/'
        outputpath = '../datasets/Fonts100/finetune/' + fontname + '/'
        mkdir(outputpath)
        img = PreProcess(inputpath + str(i)+'.png')
        cv2.imwrite(outputpath + str(i)+'.png',img)


