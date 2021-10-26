# Copyright Nima Rafiee.  Rafiee.nima@gmail.com
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
import argparse

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
    return features, labels.long().cuda(non_blocking=True)


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
        feature_list.append(feats)            
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
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=10):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
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
#             transforms.Resize([img_size, img_size], interpolation=Image.BICUBIC),              
            transforms.Resize([vit_img_size, vit_img_size], interpolation=Image.BICUBIC),             
#             transforms.RandomResizedCrop(img_size, 
#                                          scale=crops_scale,                      
#                                          interpolation=Image.BICUBIC),  
#             transforms.RandomApply(
#                 [transforms.ColorJitter(brightness=0.4, contrast=0.4, 
#                                         saturation=0.2, hue=0.1)], p=0.8
#             ),
#             transforms.RandomHorizontalFlip(p=0.5),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RotDouble head')
    parser.add_argument('--batch_size_per_gpu', default=32, type=int, help='Per-GPU batch-size')
    parser.add_argument('--local_view', default=False, type=bool,)
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--vit_image_size', type=int, default=256, help="""image size that enters vit; 
        must match with patch_size: num_patches = (vit_image_size/patch_size)**2""")
    parser.add_argument('--image_size', type=int, default=32)    
    parser.add_argument('--crops_scale', type=float, nargs='+', default=(0.9, 1.), 
                        help="""Scale range of the cropped image before resizing, relatively to the origin image. Used for large global view cropping.""")
    parser.add_argument('--crops_number', type=int, default=1, 
                        help="""Number of local views to generate. Set this parameter to 0 to disable multi-crop training.""")        
    parser.add_argument('--knn_temperature', default=0.04, type=float,
        help='Temperature used in the voting coefficient')
    
    parser.add_argument('--reduce_train', default=False , type=bool,
        help='Apply Kmean clustering to reduce train samples')  
    parser.add_argument('--num_clusters', default=40 , type=int,
        help='Number of clusters to which the train samples will be reduced')     

    parser.add_argument('--pretrained_weights', default='', 
                        type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, 
                        type=utils.bool_flag, 
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
#     parser.add_argument('--dump_features', default=None,
#         help='Path where to save computed features, empty for no saving')
#     parser.add_argument('--load_features', default=None, help="""If the features have
#         already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:23459", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, 
                        help="Please ignore and do not set this argument.")
    parser.add_argument("--master_port", default=23459, type=int, 
                        help="master port for ddp")    
    parser.add_argument('--data_path', default='/home/shared/DataSets/', type=str)
    parser.add_argument('--train_dataset', default='cifar10', type=str)
    parser.add_argument('--extra_tag', default='_', type=str)

   #python -m torch.distributed.launch --master_port 8993 --nproc_per_node=1 eval_checkpoints.py --train_dataset=cifar10 --pretrained_weights=checkpoints/... --extra_tag=InDistNeg

    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    

    dataset_list = ['cifar10', 'cifar100', 'svhn', 
        'imagenet30', 'tiny_imagenet', 'lsun',  'stl10', 'places365', 'places365_b', 'texture']
    
    num_crops_list = [1]  ## for more crops need to change augmentation 
    num_neighbour_list = [-1]
    temperature_list = [0.04] 
    

    
    #chkpt_list = [{args.train_dataset: 'checkpoints/checkpoint_cifar100.pth'}]
    ood_result_lst = []
    knn_result_lst = []
    with open(f'{args.train_dataset}_results.txt', 'a+') as file:
        file.write(f'\n\n\n################# Checkpoint: {args.pretrained_weights} ################\n\n')
        file.writelines([f'{key}: {value}\n' for key, value in vars(args).items()])
        file.write('\n\n\n')
        chkpt_name = args.train_dataset + args.extra_tag
        chkpt_path = args.pretrained_weights
        in_ds = args.train_dataset
        knn_dict = {'chkpt_name': chkpt_name, 'train_ds': in_ds}
        print(f"#########. Creating Pipline for {in_ds} ")
        model = load_model(args, chkpt_path)
        for num_crop in num_crops_list:
            train_features, train_labels = extract_feature_pipeline(args, model, in_ds, 
                                                      train=True, crops_number=num_crop)

            if args.reduce_train:
                print(f"######## Applying Kmeans with {args.num_clusters} clusters on Training Features")
                _, train_features = kmeans(X=ftrain, 
                                    num_clusters=args.num_clusters, 
                                    distance='cosine', 
                                    device=torch.device('cuda'))
                print(f"Trean Feature Size After Clustering {train_features.size()}")
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
                                                k, args.knn_temperature, 
                                                num_classes=n_cls)
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
                        print(f"######## Calculating AUROC for Num Neighbour {k} and T {T} ######")
                        scores_in = OOD_classifier(train_features, test_features, k, T, num_crop, args)
                        scores_out = OOD_classifier(train_features, ood_features, k, T, num_crop, args)
                        labels = torch.cat((torch.ones(scores_in.size(0)), 
                                            torch.zeros(scores_out.size(0))))
                        scores = torch.cat((scores_in, scores_out))
                        auroc = roc_auc_score(labels.numpy(), scores.cpu().numpy())
                        conf_dict['chkpt_name'] = chkpt_name
                        conf_dict['in_ds'] = in_ds
                        conf_dict['ood_ds'] = ood_ds
                        conf_dict['num_crop'] = num_crop
                        conf_dict['num_neighbour'] = k
                        conf_dict['T'] = T
                        conf_dict['AUROC'] = auroc
                        ood_result_lst.append(conf_dict)
                        print(f"AUROC: {auroc}")
                        file.write(f'\t\t {in_ds} \t\t {ood_ds}({ood_features.size(0)}) \t\t\t\t\t  {auroc} \n')
            file.write('___________________________ Done _______________________________')
                            
    ood_df = pd.DataFrame(ood_result_lst)
    knn_df = pd.DataFrame(knn_result_lst)
    ood_df.to_csv(f'ood_{chkpt_name}_results_df.csv', index=False)
    knn_df.to_csv(f'knn_{chkpt_name}_results_df.csv', index=False)
                            
                            
    dist.barrier()
