#!/usr/bin/python
# -*- encoding: utf-8 -*-

from models.model_stages import BiSeNet

import torch
import torch.nn.functional as F

import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms

# Some parameter setting!!!!!
scale = 0.75

input_img = './data/1.png'
output_pth = './result'
output_color_mask = 'color_mask.png'
output_composited = 'composited.png'

# palette
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

# get data
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
image = Image.open(input_img).convert('RGB')
img_tensor = img_transform(image)

# get net
net = BiSeNet(backbone='STDCNet1446', n_classes=19,
 use_boundary_2=False, use_boundary_4=False, 
 use_boundary_8=True, use_boundary_16=False, 
 use_conv_last=False)
net.load_state_dict(torch.load('./checkpoints/STDC2-Seg/model_maxmIOU75.pth'))
net.cuda()
net.eval()

# predict
with torch.no_grad():
    img = img_tensor.unsqueeze(0).cuda()
    N, C, H, W = img.size()
    new_hw = [int(H*scale), int(W*scale)]

    img = F.interpolate(img, new_hw, mode='bilinear', align_corners=True)

    logits = net(img)[0]
  
    logits = F.interpolate(logits, size=(H, W),
            mode='bilinear', align_corners=True)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)

# colorize and save image
pred = np.asarray(pred.cpu().squeeze(0), dtype=np.uint8)
colorized = Image.fromarray(pred)
colorized.putpalette(palette)
colorized.save(os.path.join(output_pth, output_color_mask))

# composite input image and colorize image
prediction_pil = colorized.convert('RGB')
composited = Image.blend(image, prediction_pil, 0.4)
composited.save(os.path.join(output_pth, output_composited))
