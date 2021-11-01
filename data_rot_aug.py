import random
import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import utils


def rotate(x):
    angle = random.choice([90, 180, 270])
    x = TF.rotate(x, angle)
    return x

def rand_rotate(x):
    dict_rot = {90:1, 180:2 ,270:3}
    angle = random.choice([90, 180, 270])
    x = TF.rotate(x, angle)
    return x, dict_rot[angle]



class DatasetRotationWrapper(Dataset):
    """
    returns an extra binary label: 0 for non-rotated and 1 for rotated images
    """

    def __init__(self, image_size, vit_image_size, global_crops_scale, local_crops_scale, local_crops_number,
                    in_dist='cifar10', data_path='/home/shared/DataSets/'):
        super().__init__()

        self.dataset = utils.get_train_dataset(in_dist, data_path)
        
        self.create_transforms(image_size, vit_image_size, global_crops_scale, local_crops_scale, local_crops_number)

    def create_transforms(self, image_size, vit_image_size, global_crops_scale, local_crops_scale, local_crops_number):
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
        # first global crop
        self.global_transfo1 = transforms.Compose([
            # to stop shortcut learning
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0, image_size=vit_image_size),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1, image_size=vit_image_size),
            utils.Solarization(0.2),
            normalize,
        ])
        # neg first global crop
        self.global_transfo1_neg = transforms.Compose([
            self.global_transfo1,
            rotate
        ])
        # neg second global crop
        self.global_transfo2_neg = transforms.Compose([
            self.global_transfo2,
            rotate
        ])

        self.local_transfo = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size // 2, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5, image_size=vit_image_size),
            normalize,
        ])
        # neg transformation for the local small crops
        self.local_transfo_neg = transforms.Compose([
            self.local_transfo,
            rotate,
        ])
        self.local_crops_number = local_crops_number

        # mock data init
        self.get_crops(TF.to_pil_image(torch.ByteTensor(vit_image_size, vit_image_size, 3).random_().numpy()))

    def get_crops(self, image):
        crops = []
        self.crops_freq_teacher = []
        self.crops_freq_student = []
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
        
        n_neg_crops = len(crops) - n_pos_crops
        rot_labels = [1] * n_pos_crops + [0] * n_neg_crops
        return crops, rot_labels

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # transform
        crops, rot_labels = self.get_crops(img)
        labels = [label] * len(crops)
        return crops, labels, rot_labels

    def __len__(self):
        return len(self.dataset)

class DatasetRotationWrapperPred(DatasetRotationWrapper):
    """
    returns an extra rot label: 0,1,2,3 for 0,90,180,270 rot
    """
    def __init__(self, image_size, vit_image_size, global_crops_scale, local_crops_scale, local_crops_number, in_dist='cifar10'):
        super().__init__(image_size, vit_image_size, global_crops_scale, local_crops_scale, local_crops_number, in_dist)

    def get_crops(self, image):
        crops = []
        self.crops_freq_teacher = []
        self.crops_freq_student = []
        angles_classes = []
        # pos crops
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        angles_classes.extend([0,0])
        
        self.crops_freq_teacher.append(len(crops))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
            angles_classes.append(0)
        self.crops_freq_student.append(len(crops))
        n_pos_crops = len(crops)

        # in-dist neg crops
        
        flip = random.choice([0, 1])
        if flip:
            view = self.global_transfo1(image)
        else:
            view = self.global_transfo2(image)
        
        view, angle = rand_rotate(view)
        crops.append(view)
        angles_classes.append(angle)

        self.crops_freq_teacher.append(len(crops) - n_pos_crops)
        for _ in range(self.local_crops_number):
            view, angle = rand_rotate(self.local_transfo(image))
            crops.append(view)
            angles_classes.append(angle)
        
        self.crops_freq_student.append(len(crops) - n_pos_crops)
        
        return crops, angles_classes
    