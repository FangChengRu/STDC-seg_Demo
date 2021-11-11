#!/usr/bin/python
# -*- encoding: utf-8 -*-

from models.model_stages import BiSeNet
from cityscapes import CityScapes

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import numpy as np
from tqdm import tqdm
from PIL import Image

class MscEvalV0(object):

    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        ## evaluate
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:

            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]

            imgs = imgs.cuda()

            N, C, H, W = imgs.size()
            new_hw = [int(H*self.scale), int(W*self.scale)]

            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

            logits = net(imgs)[0]
  
            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
        return preds


# palette
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

# colorize and save image
def save_pred(preds, sv_path, name):
    preds = np.asarray(preds.cpu().squeeze(0), dtype=np.uint8)
    for i in range(preds.shape[0]):
        save_img = Image.fromarray(preds[i])
        save_img.putpalette(palette)
        save_img.save(os.path.join(sv_path, f'{name[i]}.png'))

# output image name list
name = []       
for i in range(500):
    name.append(i)

# load datas
batchsize = 4
n_workers = 4
dsval = CityScapes('./data', mode='val')
dl = DataLoader(dsval,
                batch_size = batchsize,
                shuffle = False,
                num_workers = n_workers,
                drop_last = False)


net = BiSeNet(backbone='STDCNet1446', n_classes=19,
 use_boundary_2=False, use_boundary_4=False, 
 use_boundary_8=True, use_boundary_16=False, 
 use_conv_last=False)
net.load_state_dict(torch.load('./checkpoints/STDC2-Seg/model_maxmIOU75.pth'))
net.cuda()
net.eval()

with torch.no_grad():
    single_scale = MscEvalV0(scale=0.75)
    preds = single_scale(net, dl, 19)
save_pred(preds, './result', name)


        
'''
if __name__ == "__main__":
    log_dir = 'evaluation_logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setup_logger(log_dir)
    
    #STDC1-Seg50 mIoU 0.7222
    # evaluatev0('./checkpoints/STDC1-Seg/model_maxmIOU50.pth', dspth='./data', backbone='STDCNet813', scale=0.5, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    #STDC1-Seg75 mIoU 0.7450
    # evaluatev0('./checkpoints/STDC1-Seg/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet813', scale=0.75, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)


    #STDC2-Seg50 mIoU 0.7424
    # evaluatev0('./checkpoints/STDC2-Seg/model_maxmIOU50.pth', dspth='./data', backbone='STDCNet1446', scale=0.5, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    #STDC2-Seg75 mIoU 0.7704
    evaluatev0('./checkpoints/STDC2-Seg/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75, 
    use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
''' 