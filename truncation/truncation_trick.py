import vision_transformer as vits
import sys
import utils
import torch
from PIL import Image
from torch import nn
from torchvision import datasets
from torchvision import transforms
from vision_transformer import DINOHead

from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

import numpy as np

class IndexData(Dataset):
    def __init__(self, args, transform, train=True, label=False, type="cifar10"):
        super().__init__()
        self.label = label
        mode = 'test' if not train else 'train'
        if type == "cifar10":
            self.dataset = datasets.CIFAR10(root=args.data_path,
                                            train=train,
                                            download=True, transform=transform)
        elif type == "cifar100":
            self.dataset = datasets.CIFAR100(root=args.data_path,
                                             train=train,
                                             download=True, transform=transform)
        elif type == "svhn":
            self.dataset = datasets.SVHN(root=args.data_path,
                                         split=mode,
                                         download=True, transform=transform)
        elif type == "stl10":
            self.dataset = datasets.STL10(root=args.data_path,
                                          split=mode,
                                          download=True, transform=transform)
        elif type == "places365":
            self.dataset = datasets.Places365(root=args.data_path,
                                              split='val' if not train else 'train-standard',
                                              download=False, small=True,
                                              transform=transform)
        elif type == "lsun":
            self.dataset = datasets.LSUN(root=args.data_path,
                                         classes=mode,
                                         transform=transform)

        elif type == 'tiny_imagenet':
            self.dataset = datasets.ImageFolder(root=args.data_path + f'tiny-imagenet-200/{mode}/',
                                                transform=transform)
        elif type == 'imagenet30':
            self.dataset = datasets.ImageFolder(root=args.data_path + f'ImageNet30/{mode}/',
                                                transform=transform)
        elif type == 'texture':
            self.dataset = datasets.ImageFolder(root=args.data_path + f'dtd_test',
                                                transform=transform)

        else:
            print(f"{type} does not exit")

    def __getitem__(self, idx):
        img, lab = self.dataset[idx]
        if self.label == True:
            return img, lab
        else:
            return img, idx

    def __len__(self):
        return len(self.dataset)


class DataAugmentation(object):
    def __init__(self, args, crops_number=1):

        self.crops_number = crops_number
        self.local_view = args.local_view
        img_size = args.vit_image_size
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.transform_global = transforms.Compose([
            transforms.Resize([img_size, img_size],
                              interpolation=Image.BICUBIC),
            normalize,
        ])

        self.transform_local = transforms.Compose([
            transforms.Resize(img_size,
                              interpolation=InterpolationMode.BICUBIC),
            transforms.FiveCrop(img_size // 2),
            transforms.Lambda(lambda crops: [normalize(crop) for crop in crops])
        ])

    def __call__(self, image):
        if self.local_view:
            crops = [self.transform_global(image) for i in range(self.crops_number)]
        else:
            crops = [self.transform_global(image) for i in range(self.crops_number)]
        return crops


def extract_feature_pipeline(args, model, ds_name,
                             train=True, crops_number=1,
                             normalise=True, label=True):
    # ============ preparing data ... ============
    transform = DataAugmentation(args, crops_number)
    dataset = IndexData(args, transform, train=train, label=label, type=ds_name)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ============ extract features ... ============
    features, labels = extract_features(model, data_loader, args, crops_number)
    print(f"\n All features.shape {features.shape}")
    if args.use_cuda:
        labels = labels.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
    if normalise and utils.get_rank() == 0:
        features = nn.functional.normalize(features, dim=1, p=2)
    print(f"Feature Size for {ds_name} {'Test' if not train else 'Train'}:{features.size()}")
    print(f"Lable Size for {ds_name} {'Test' if not train else 'Train'}:{labels.size()}")
    return features, labels.long()





@torch.no_grad()
def extract_features(model, data_loader, args, crops_number=1, test=True):
    feature_list = []
    label_list = []
    
    for i, (x_list, labels) in enumerate(data_loader):
        bs, c, w, h = x_list[0].size()
        
        crop_lst = [x.unsqueeze(1).cuda(non_blocking=True) for x in x_list]
        x = torch.cat(crop_lst, dim=1)
        x = x.view(bs * crops_number, c, w, h)
        feats = model(x)
        feature_list.append(feats.cpu())
        label_list.append(labels.cpu())


        

    feature = torch.cat(feature_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    return feature, labels



def load_model(args, chkpt_path, remove_head=True):
    # ============ building network ... ============
    if "vit" in args.arch:
        num_classes = 0
        model = vits.__dict__[args.arch](patch_size=args.patch_size,
                                         img_size=[args.vit_image_size],
                                         num_classes=num_classes)
        if not remove_head:
            model = DINOWrapper(model, DINOHead(
                model.embed_dim,
                args.pretrained_out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer))
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    
    model.cuda() if torch.cuda.is_available() else model.cpu()
    
    utils.load_pretrained_weights(model, chkpt_path,
                                  args.checkpoint_key,
                                  args.arch, args.patch_size, remove_head=remove_head)
    model.eval()
    return model


def truncationTrick(args, chkpt_path, in_ds, num_crop, occupied_classes=True, load=None,temp=0.01):

    if load is None:
        # load full model with head
        model = load_model(args, chkpt_path, remove_head=False)
        print('\n Extracting probabilities...\n')
        train_features, _ = extract_feature_pipeline(args, model, in_ds, normalise=False,
                                                                train=True, crops_number=num_crop)

        torch.save(train_features, './train_feats_cifar10.obj')
    else:
        train_features = torch.load(load)


    occ_classes = get_occupied_classes(train_features, temp=temp)

    train_features_norm = norm_temp_softmax_features(train_features, temp)

    # find median feature index from rows

    (_, indices) = torch.median(train_features_norm, dim=0) #check
    print("indices shape",indices.shape)
    unique_indices = torch.unique(indices, sorted=True)
    print("indices unique shape:", unique_indices.shape) # 56

    if occupied_classes:
        return 

    train_features_final, train_labels_final = extract_feature_pipeline(args, model.backbone, in_ds, normalise=True,
                                                            train=True, crops_number=num_crop)
    
    print('Extracted train_features from feature space:', train_features.shape)
    x_medians = train_features[indices,...]
    print("medians shape:",x_medians.shape)
    
    # find distance from query to all class x_medians vectors

    # calculated per class prob weighted sum of class


    # remove train_features

    return train_features_final, train_labels_final






class DINOWrapper(nn.Module):

    def __init__(self, backbone, head):
        super(DINOWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        y = self.backbone(x)
        return self.head(y)

import torch
import numpy as np


def norm_temp_softmax_features(train_features, temp):
    center = train_features.mean(dim=0)
    return torch.nn.functional.softmax((train_features-center)/temp,dim=1)

def get_occupied_classes(train_features, temp=0.07, plot=True):
    train_features_norm = norm_temp_softmax_features(train_features, temp)
    
    # class probabilities
    class_weights = train_features_norm.mean(dim=0)

    occupied_classes = (class_weights>class_weights.mean()).sum()
    if plot:
        import matplotlib.pyplot as plt
        sorted_np_array = np.sort(class_weights.cpu().view(-1).numpy())[-occupied_classes:]
        ids = [i for i in range(sorted_np_array.shape[0])]
        _ = plt.bar(ids,sorted_np_array)
        plt.savefig('fig_occupied_classes_box_plot.png')
    
    return occupied_classes