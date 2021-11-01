import os

import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch import linalg
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from torchvision import models as torchvision_models
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import pandas as pd 
import utils
import vision_transformer as vits    

from main_train import get_args_parser


def eval_routine(args, train_dataset, pretrained_weights, overwrite_args=False):
    args.checkpoint_key="teacher"
    args.crops_scale=(0.9, 1.)
    args.data_path = '/home/shared/DataSets/'
    args.local_view =False
    
    if overwrite_args:
        args.patch_size=16
        args.vit_image_size=256
        args.image_size = 32
        
    eval_checkpoints(train_dataset=train_dataset, 
            pretrained_weights=pretrained_weights, args=args)


def eval_checkpoints(train_dataset, pretrained_weights, args, extra_tag='res', knn_temperature=0.04, outdir='./results/'):
    dataset_list = ['cifar10', 'cifar100', 'svhn', 'imagenet30', 'lsun', 
                    'tiny_imagenet',  'stl10', 'places365', 'places365_b', 'texture']

    Path(outdir).mkdir(parents=True, exist_ok=True)

    num_crops_list = [1]  ## for more crops need to change augmentation 
    num_neighbour_list = [-1]
    temperature_list = [0.04] 
    

    ood_result_lst = []
    knn_result_lst = []

    chkpt_name = train_dataset + extra_tag
    chkpt_path = pretrained_weights
    txt_name = f"{train_dataset}_results.txt"
    out_file = os.path.join(outdir, txt_name)
    out_csv1 = os.path.join(outdir, f'ood_{chkpt_name}_results_df.csv')
    out_csv2 = os.path.join(outdir, f'knn_{chkpt_name}_results_df.csv')
    with open(out_file, 'a+') as file:
        file.write(f'\n\n\n################# Checkpoint: {pretrained_weights} ################\n\n')
        file.writelines([f'{key}: {value}\n' for key, value in vars(args).items()])
        file.write('\n\n\n')
        
        in_ds = train_dataset
        knn_dict = {'chkpt_name': chkpt_name, 'train_ds': in_ds}
        print(f"#########. Creating Pipline for {in_ds} ")
        model = load_model(args, chkpt_path)
        for num_crop in num_crops_list:
            train_features, train_labels = extract_feature_pipeline(args, model, in_ds, 
                                                    train=True, crops_number=num_crop)
            
            dist.barrier()
            if args.use_cuda:
                train_labels = train_labels.cuda(non_blocking=True)
            
            test_features, test_labels = extract_feature_pipeline(args, model, in_ds, 
                                                    train=False)

            if num_crop == 1:
                file.write("___________KNN Results__________________________________\n\n")
                for k in [10, 20]: 
                    if in_ds in ['cifar10', 'svhn']:
                        n_cls = 10
                    elif in_ds in ['cifar100']:
                        n_cls = 100
                    elif in_ds in ['imagenet30']:
                        n_cls = 30                            
                    else:
                        raise NotImplemented                             
                    top1, top5 = knn_classifier(train_features,train_labels,
                                                test_features, test_labels, 
                                                k, knn_temperature, 
                                                num_classes=n_cls, use_cuda=args.use_cuda)
                    knn_dict[f'{k}NN_Top1'] = top1
                    knn_dict[f'{k}NN_Top5'] = top5
                    file.write(f'{k}NN_Top1 \t\t\t {top1}\n')
                knn_result_lst.append(knn_dict)
                
            file.write("\n\n___________OOD Results__________________________________\n\n")  
            file.write("\t\t In \t\t  OOD(Num Samples)  \t\t\t   AUROC \n")
            for ood_ds in dataset_list:
                if ood_ds == in_ds:
                    continue
                ood_features, _ = extract_feature_pipeline(args, model, ood_ds, train=False)    
                for k in num_neighbour_list:
                    for T in temperature_list:
                        conf_dict = {}
                        print(f"\n\n ######## Calculating AUROC for {ood_ds} || Num Neighbour {k} and T {T} ######\n\n")
                        scores_in = OOD_classifier(train_features, test_features, k, T, num_crop, args)
                        scores_out = OOD_classifier(train_features, ood_features, k, T, num_crop, args)
                        labels = torch.cat((torch.ones(scores_in.size(0)), 
                                            torch.zeros(scores_out.size(0))))
                        scores = torch.cat((scores_in, scores_out))
                        dist.barrier()

                        if utils.is_main_process():
                            auroc = roc_auc_score(labels.numpy(), scores.cpu().numpy())
                            conf_dict['chkpt_name'] = chkpt_name
                            conf_dict['in_ds'] = in_ds
                            conf_dict['ood_ds'] = ood_ds
                            conf_dict['num_crop'] = num_crop
                            conf_dict['num_neighbour'] = k
                            conf_dict['T'] = T
                            conf_dict['AUROC'] = auroc
                            ood_result_lst.append(conf_dict)
                            print(f"\n\nAUROC {ood_ds}: {auroc}\n\n")
                            file.write(f'\t\t {in_ds} \t\t {ood_ds}({ood_features.size(0)}) \t\t\t\t\t  {auroc} \n')
            file.write('___________________________ Done _______________________________')
    if utils.is_main_process():                        
        ood_df = pd.DataFrame(ood_result_lst)
        knn_df = pd.DataFrame(knn_result_lst)
        ood_df.to_csv(out_csv1, index=False)
        knn_df.to_csv(out_csv2, index=False)

    dist.barrier()


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
    if normalise and utils.get_rank() == 0:
        features = nn.functional.normalize(features, dim=1, p=2)
    print(f"Feature Size for {ds_name} {'Test' if not train else 'Train'}:{features.size()}")
    print(f"Lable Size for {ds_name} {'Test' if not train else 'Train'}:{labels.size()}")
    return features, labels.long()


def load_model(args, chkpt_path):
    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size,
                                         img_size = [args.vit_image_size],         
                                         num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, chkpt_path, 
                                  args.checkpoint_key, 
                                  args.arch, args.patch_size)
    model.eval()
    return model 

@torch.no_grad()
def extract_features(model, data_loader, args, crops_number=1):
    feature_list = []
    label_list = []
    for i, (x_list, labels) in enumerate(data_loader):
        bs, c, w, h = x_list[0].size()  
        crop_lst = [x.unsqueeze(1).cuda(non_blocking=True) for x in x_list]
        x = torch.cat(crop_lst, dim=1)
        x = x.view(bs*crops_number, c, w, h)
        feats = model(x)
        
        if args.use_cuda:
            feature_list.append(feats)
        else:
            feature_list.append(feats.to('cpu'))
            
        label_list.append(labels)
    feature = torch.cat(feature_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    
    return feature, labels


@torch.no_grad()
def OOD_classifier(train_features, test_features, k, T, cn ,args):
    train_features = train_features.t()
    batch_size =  1000
    cos_sim_lst = []
    num_test_feat = test_features.size(0)
    for strat_idx in range(0, num_test_feat, batch_size):
        end_idx = min((strat_idx + batch_size), num_test_feat)
        curr_test_features = test_features[strat_idx : end_idx]   
        curr_bs = curr_test_features.size(0)
        similarity = torch.mm(curr_test_features, train_features)
        if k != -1:
            similarity, indices = similarity.topk(k, largest=True, sorted=True)
        if T != -1:
            similarity = (similarity - 0.1).div_(T).exp_()
        cos_sim = similarity.mean(dim=1)
        cos_sim = cos_sim.view(curr_bs, cn).mean(dim=1)
        cos_sim_lst.append(cos_sim.cpu())
    cos_sim = torch.cat(cos_sim_lst, dim=0)
    return cos_sim



@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=10, use_cuda=False):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes)
    if use_cuda:
        retrieval_one_hot = retrieval_one_hot.cuda()
    
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)
        
        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5




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
        elif type == "places365_b":
            self.dataset = datasets.Places365(root=args.data_path, 
                                          split='val' if not train else 'train-standard',
                                          download=False, small=False,
                                          transform=transform)             
            
        elif type == "lsun":
            self.dataset = datasets.LSUN( root=args.data_path, 
                                          classes=mode,
                                          transform=transform)       
            
        elif type =='tiny_imagenet':
             self.dataset = datasets.ImageFolder(root=args.data_path+f'tiny-imagenet-200/{mode}/', 
                                                 transform=transform) 
        elif type =='imagenet30':
             self.dataset = datasets.ImageFolder(root=args.data_path+f'ImageNet30/{mode}/', 
                                                 transform=transform)      
        elif type =='texture':
             self.dataset = datasets.ImageFolder(root=args.data_path+f'dtd_test', 
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
        vit_img_size = args.vit_image_size
        img_size = args.image_size
        crops_scale = args.crops_scale
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.transform_global = transforms.Compose([           
            transforms.Resize([vit_img_size, vit_img_size], interpolation=Image.BICUBIC),             

            normalize,
        ])
            
        self.transform_local = transforms.Compose([
            transforms.Resize(img_size,
                              interpolation=InterpolationMode.BICUBIC),
            transforms.FiveCrop(img_size//2),
            transforms.Lambda(lambda crops: [normalize(crop) for crop in crops])
        ])
            
         
    def __call__(self, image):
        if self.local_view:
            crops = [self.transform_global(image) for i in range(self.crops_number)]            
        else:
            crops = [self.transform_global(image) for i in range(self.crops_number)]
        return crops

