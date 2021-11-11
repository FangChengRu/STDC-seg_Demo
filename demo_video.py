#!/usr/bin/python
# -*- encoding: utf-8 -*-

from models.model_stages import BiSeNet

import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import cv2
import time

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
img = Image.open('./data/1.png').convert('RGB')
img_tensor = img_transform(img)

net = BiSeNet(backbone='STDCNet1446', n_classes=19,
 use_boundary_2=False, use_boundary_4=False, 
 use_boundary_8=True, use_boundary_16=False, 
 use_conv_last=False)
net.load_state_dict(torch.load('./checkpoints/STDC2-Seg/model_maxmIOU75.pth'))
net.cuda()
net.eval()

scale = 0.75

# load video
cap = cv2.VideoCapture("Taipei.mp4")

while(True):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
    start = time.time()

    cv2.imshow("Origin", frame)
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = img_transform(img)

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

    pred = np.asarray(pred.cpu().squeeze(0), dtype=np.uint8)
    colorized = Image.fromarray(pred)
    colorized.putpalette(palette)
    
    image = np.array(colorized.convert('RGB'))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    cv2.imshow('Realtime Segmentation', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()