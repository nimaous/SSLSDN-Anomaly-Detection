# coding=utf-8
# Copyright 2022 XXXXXX.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import time
import math
import random
import datetime
import subprocess
import itertools
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, image_size=224, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min * (image_size/224.)
        self.radius_max = radius_max * (image_size/224.)

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
        
        
def rotate(x):
    angle = random.choice([90, 180, 270])
    return TF.rotate(x, angle)        



class Identity(nn.Module):
    """
    No augmentation 
    Corresponding to ImgN in table2 
    """
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class RandomPixelPerm(nn.Module):
    """
    Random pixel permutation of a given image
    Corresponding to Pix. Perm augmentation in table2 
    """
    def __init__(self):
        super(RandomPixelPerm, self).__init__()
    def forward(self, x):
        c, w, h = x.size()
        shuffle_idx = torch.randperm(w*h)
        c0 = x[0].view(-1)[shuffle_idx].view(w, h).unsqueeze(0)
        c1 = x[1].view(-1)[shuffle_idx].view(w, h).unsqueeze(0)
        c2 = x[2].view(-1)[shuffle_idx].view(w, h).unsqueeze(0)
        output = torch.cat((c0, c1, c2), dim=0)
        return output
    
    
class Perm4(nn.Module):
    """
    Random patch permutation of a given image with splitting the image into 4 pachtes
    Corresponding to Perm-4 augmentation in table2    
    Arguments:
     shape = [c, w, h]
    
    """
    def __init__(self):
        super(Perm4, self).__init__()
    def forward(self, x):
        c, w, h = x.size()
        h_mid = int(h / 2)
        w_mid = int(w / 2)
        lst = [x[:, 0:h_mid, 0:w_mid].unsqueeze(0),                
               x[:, 0:h_mid, w_mid:].unsqueeze(0),
               x[:, h_mid:, 0:w_mid].unsqueeze(0), 
               x[:, h_mid:, w_mid:].unsqueeze(0),]
        random.shuffle(lst)
        perm = (list(itertools. permutations([0,1,2,3]))[1:])
        idx = list(perm[np.random.randint(len(perm))])
        output1 = torch.cat((lst[idx[0]], lst[idx[1]]), dim=3)
        output2 = torch.cat((lst[idx[2]], lst[idx[3]]), dim=3)
        output = (torch.cat((output1, output2), dim=2)).squeeze()
        return output
    
    
    
class Perm16(nn.Module):
    """
    Random patch permutation of a given image with splitting the image into 16 pachtes
    Corresponding to Perm-16 augmentation in table2    
    Arguments:
     shape = [c, w, h]
    
    """    
    def __init__(self):
        super(Perm16, self).__init__()
        self.num_patch = 16
    def forward(self, x):
        c, w, h = x.size()
        k = 0
        h_idx = list(torch.chunk(torch.tensor(np.arange(h)), 4))
        w_idx = list(torch.chunk(torch.tensor(np.arange(w)), 4))
        lst = []
        for i in range(len(h_idx)):
            s_h = h_idx[i][0].item()
            e_h = h_idx[i][-1].item()
            for j in range(len(w_idx)):
                s_w = w_idx[j][0].item()
                e_w = w_idx[j][-1].item()
                lst.append(x[:, s_h:e_h+1, s_w:e_w+1].unsqueeze(0))        
        random.shuffle(lst)
        idx = list(np.arange(self.num_patch))
        random.shuffle(idx)
        output1 = torch.cat((lst[idx[k]], lst[idx[k+1]], lst[idx[k+2]], lst[idx[k+3]]), dim=3)
        output2 = torch.cat((lst[idx[k+4]], lst[idx[k+5]], lst[idx[k+6]], lst[idx[k+7]]), dim=3)
        output3 = torch.cat((lst[idx[k+8]], lst[idx[k+9]], lst[idx[k+10]], lst[idx[k+11]]), dim=3)
        output4 = torch.cat((lst[idx[k+12]], lst[idx[k+13]], lst[idx[k+14]], lst[idx[k+15]]), dim=3)
        output = (torch.cat((output1, output2, output3, output4), dim=2)).squeeze()
        return output      
