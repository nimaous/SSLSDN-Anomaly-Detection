import argparse
import os
import sys
import datetime
import time
import math
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import torchvision.transforms.functional as TF
import utils

    
class DataAugmentation_Contrast(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, image_size, vit_image_size, aux=False): 
        self.aux = aux
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])


        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        
        def rotate(x):
            angle = random.choice([90, 180, 270])
            return TF.rotate(x, angle)


        # no crop
        self.no_transfo = transforms.Compose([
            transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            normalize,
        ])      

        # neg no crop
        self.no_transfo_neg = transforms.Compose([
            transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            rotate,
            normalize,
        ])      
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0, image_size=vit_image_size),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1, image_size=vit_image_size),
            utils.Solarization(0.2),
            normalize,
        ])        
        # neg first global crop
        self.global_transfo1_neg = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            rotate,
            flip_and_color_jitter,
            transforms.RandomAffine(0, translate=(0.3, 0.3)),
            utils.GaussianBlur(1.0, image_size=vit_image_size),
            normalize,
        ])
        # neg second global crop
        self.global_transfo2_neg = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            rotate,
            flip_and_color_jitter,
            transforms.RandomAffine(0, translate=(0.3, 0.3)),
            utils.GaussianBlur(0.1, image_size=vit_image_size),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_transfo = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size//2, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5, image_size=vit_image_size),
            normalize,
        ])       
        # neg transformation for the local small crops
        self.local_transfo_neg = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size//2, scale=local_crops_scale, interpolation=Image.BICUBIC),
            rotate,
            transforms.RandomAffine(0, translate=(0.3, 0.3)),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5, image_size=vit_image_size),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        # initalize crops_freq_teacher/crops_freq_student (number of crops in teacher/student output per dataset)
        img_tensor = torch.ByteTensor(vit_image_size, vit_image_size, 3).random_().numpy()
        self.get_crops(TF.to_pil_image(img_tensor))
        

    def get_crops(self, image):
        crops = []
        self.crops_freq_teacher = []
        self.crops_freq_student = []
        if self.aux == True:
            # aux crops
            flip = random.choice([0, 1])
            if flip:
                crops.append(self.global_transfo1_neg(image))
            else:
                crops.append(self.global_transfo2_neg(image))
            self.crops_freq_teacher.append(len(crops))
            for _ in range(self.local_crops_number):
                crops.append(self.local_transfo_neg(image))
            self.crops_freq_student.append(len(crops))
        else:   
            # pos crops
            crops.append(self.global_transfo1(image))
            crops.append(self.global_transfo2(image))
            self.crops_freq_teacher.append(len(crops))
            for _ in range(self.local_crops_number):
                crops.append(self.local_transfo(image))
            self.crops_freq_student.append(len(crops))
            n_pos_crops = len(crops)
            # in-dist neg crops 
            flip = random.choice([0, 1])
            if flip:
                crops.append(self.global_transfo1_neg(image))
            else:
                crops.append(self.global_transfo2_neg(image))
            self.crops_freq_teacher.append(len(crops) - n_pos_crops)
            for _ in range(self.local_crops_number):
                crops.append(self.local_transfo_neg(image))       
            self.crops_freq_student.append(len(crops) - n_pos_crops)
        return crops
         
        
    def __call__(self, image):
        return self.get_crops(image)


                                          


