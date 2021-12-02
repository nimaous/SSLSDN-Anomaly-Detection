# Copyright XXXXXXXXXXXX
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


import sys
import argparse

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch import nn
from torch import linalg
from torchvision import datasets
from torchvision import transforms
from torchvision import models as torchvision_models
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

from PIL import Image
from sklearn.metrics import roc_auc_score
import utils
import vision_transformer as vits


def extract_feature_pipeline(args, path, model, ds_name, 
                             train=True, crops_number=1,
                             normalise=True, label=True):
    # ============ preparing data ... ============
    transform = DataAugmentation(args, crops_number)
    dataset = IndexData(path, transform, train=train, label=label, type=ds_name)
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
    #features, labels = extract_features(model, data_loader)        
    if normalise and utils.get_rank() == 0:
        features = nn.functional.normalize(features, dim=1, p=2)
    print(f"Feature Size for {ds_name} {'Test' if not train else 'Train'}:{features.size()}")
    #print(f"Lable Size for {ds_name} {'Test' if not train else 'Train'}:{labels.size()}")
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
    def __init__(self, path, transform, train=True, label=False, type="cifar10"):
        super().__init__()
        self.label = label  
        mode = 'test' if not train else 'train'
        if type == "cifar10":
            self.dataset = datasets.CIFAR10(root=path,
                                train=train,
                                download=True, transform=transform)
        elif type == "cifar100":
            self.dataset = datasets.CIFAR100(root=path,
                                train=train,
                                download=True, transform=transform)
        elif type == "svhn":
            self.dataset = datasets.SVHN(root=path,
                                split=mode,
                                download=True, transform=transform)
        elif type == "stl10":
            self.dataset = datasets.STL10(root=path, 
                                          split=mode,
                                          download=True, transform=transform)
        elif type == "places365":
            self.dataset = datasets.Places365(root=path, 
                                          split='val' if not train else 'train-standard',
                                          download=False, small=True,
                                          transform=transform)  
        elif type == "places365_b":
            self.dataset = datasets.Places365(root=path, 
                                          split='val' if not train else 'train-standard',
                                          download=False, small=False,
                                          transform=transform)             
        elif type == "lsun":
            self.dataset = datasets.LSUN( root=path, 
                                          classes=mode,
                                          transform=transform)        
        elif type =='tiny_imagenet':
             self.dataset = datasets.ImageFolder(root=path+f'tiny-imagenet-200/{mode}/', 
                                                 transform=transform) 
        elif type =='imagenet30':
             self.dataset = datasets.ImageFolder(root=path+f'ImageNet30/{mode}/', 
                                                 transform=transform)      
        elif type =='texture':
             self.dataset = datasets.ImageFolder(root=path+f'dtd_test', 
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
            transforms.Resize([img_size, img_size], interpolation=Image.BICUBIC),              
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
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
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:23459", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, 
                        help="Please ignore and do not set this argument.")
    parser.add_argument("--master_port", default=23459, type=int, 
                        help="master port for ddp")    
    parser.add_argument('--in_data_path', default='.', type=str)
    parser.add_argument('--ood_data_path', default='.', type=str)
    parser.add_argument('--in_dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--ood_dataset', default='cifar100', type=str,
                       choices=['texture', 'cifar100', 'svhn', 'imagenet30', 'lsun', 
                    'tiny_imagenet', 'cifar10',  'stl10', 'places365'])    


    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    chkpt_name = args.in_dataset + args.pretrained_weights #args.extra_tag
    knn_dict = {'chkpt_name': chkpt_name, 'train_ds': args.in_dataset}
    print(f"#########. Creating Pipline for {args.in_dataset} ")
    model = load_model(args, args.pretrained_weights)
    train_features, train_labels = extract_feature_pipeline(args, args.in_data_path,
                                                            model, args.in_dataset, 
                                                            train=True)
    if args.reduce_train:
        print(f"######## Applying Kmeans with {args.num_clusters} clusters on Training Features")
        _, train_features = kmeans(X=ftrain, 
                            num_clusters=args.num_clusters, 
                            distance='cosine', 
                            device=torch.device('cuda'))
        print(f"Trean Feature Size After Clustering {train_features.size()}")
    test_features, test_labels = extract_feature_pipeline(args, args.in_data_path,
                                                          model, args.in_dataset, train=False)

    for k in [10, 20]: 
        if args.in_dataset in ['cifar10']:
            n_cls = 10
        elif args.in_dataset in ['cifar100']:
            n_cls = 100
        else:
            raise NotImplemented                             
        top1, top5 = knn_classifier(train_features,train_labels,
                                    test_features, test_labels, 
                                    k, args.knn_temperature, 
                                    num_classes=n_cls)
        knn_dict[f'{k}NN_Top1'] = top1
        knn_dict[f'{k}NN_Top5'] = top5
                
    ood_features, _ = extract_feature_pipeline(args, args.ood_data_path, model, 
                                               args.ood_dataset, train=False)    

    scores_in = OOD_classifier(train_features, test_features, -1, 0.04, 1, args)
    scores_out = OOD_classifier(train_features, ood_features, -1, 0.04, 1, args)
    labels = torch.cat((torch.ones(scores_in.size(0)), 
                        torch.zeros(scores_out.size(0))))
    scores = torch.cat((scores_in, scores_out))
    auroc = roc_auc_score(labels.numpy(), scores.cpu().numpy())                       
    dist.barrier()
    print("\n\n")
    print(f"KNN Accuracy: \n")
    print(f"10NN_Top1: {knn_dict['10NN_Top1']}")        
    print(f"AUROC Scores: \n")
    print(f" in-dist {args.in_dataset} / ood {args.ood_dataset} = {auroc}")  
    print("\n\n")
