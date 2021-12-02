# Copyright (c) XXXXXXXXXXXX
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
import data_utils as d_utils
import vision_transformer as vits
from vision_transformer import DINOHead



torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    
    # Train data and negative Sampling paramteres
    parser.add_argument('--train_data', default="cifar10", type=str, 
                       choices=['cifar10', 'cifar100'], help='traing data type')
    parser.add_argument('--train_data_path', type=str,
        help='Please specify path to the training data.')       
    parser.add_argument('--auxiliary_data_path', type=str,
        help='Please specify path to the ImageNet auxiliary data.')            
    parser.add_argument('--indist_only', default=False, type=utils.bool_flag,
        help="Whether to use only negative sampling from in-distribution data")
    parser.add_argument('--aux_only', default=False, type=utils.bool_flag,
        help="Whether to use only negative sampling from auxiliary dataset")    
    parser.add_argument('--neg_aug', type=str, default="rot", choices=['rot', 'perm4', 'perm16', 'pixel_perm', 'rot_360', 'img_n'],
        help='Negative sample augmentation')                   
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--warmup_contrast_temp', default=0.12, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--contrast_temp', default=0.24, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_contrast_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')


    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--vit_image_size', type=int, default=256, help="""image size that enters vit; 
        must match with patch_size: num_patches = (vit_image_size/patch_size)**2""")
    parser.add_argument('--image_size', type=int, default=256, help="""image size of in-distibution data. 
        negative samples are first resized to image_size and then inflated to vit_image_size. This
        ensures that aux samples have same resolution as in-dist samples""")



    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============

    transform = DataAugmentation_Contrast(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.image_size,
        args.vit_image_size, 
        aux = False,
    )

    transform_aux = DataAugmentation_Contrast(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.image_size,
        args.vit_image_size,      
        aux = True,
    )
    
    if args.train_data == 'cifar10':
        dataset = datasets.CIFAR10(root=args.train_data_path, download=True,
                                                      transform=transform)
    elif args.train_data == 'cifar100':
        dataset = datasets.CIFAR100(root=args.train_data_path, download=True,
                                    transform=transform)
    else:
        raise NotImplemented 
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"In-distribution Data loaded: there are {len(dataset)} images.")


    dataset_aux = datasets.ImageFolder(args.auxiliary_data_path,
                                       transform=transform_aux)  
    sampler_aux = torch.utils.data.DistributedSampler(dataset_aux, shuffle=True)
    data_loader_aux = torch.utils.data.DataLoader(
        dataset_aux,
        sampler=sampler_aux,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Aux Data loaded: there are {len(dataset_aux)} images.")
    print("len dataloader",len(data_loader))
    
    
    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            img_size=[args.vit_image_size],
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](
            img_size = [args.vit_image_size],
            patch_size=args.patch_size,
        )
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit"):
        student = torch.hub.load('facebookresearch/xcit', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
   
    dino_loss = DINOLossNegCon(
        out_dim=args.out_dim,
        batchsize=args.batch_size_per_gpu,  # total number of crops = 2 global crops  + local_crops_number 
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        indist_only=args.indist_only, 
        aux_only=args.aux_only        
    ).cuda()

    

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, data_loader_aux, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        torch.set_printoptions(profile="full")
        if epoch % 20 == 0:
            print("probs pos", torch.topk(dino_loss.probs_pos * 100, 200)[0])
            print("probs neg", torch.topk(dino_loss.probs_neg * 100, 200, largest=False)[0])
        torch.set_printoptions(profile="default")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader, data_loader_aux,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        try:
            images_aux, _ = next(iter_aux)    # domain knowlege e.g. images of natural objects 
        except:  
            iter_aux = iter(data_loader_aux)
            images_aux, _ = next(iter_aux)
       
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration        
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        images_aux = [im.cuda(non_blocking=True) for im in images_aux]

        
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # number of crops per dataset: e.g. crops_freq_teacher = [3,1,1] => 3 x pos , 1 x in-dist neg , 1 x aux neg
            crops_freq_teacher = data_loader.dataset.transform.crops_freq_teacher + \
                                        data_loader_aux.dataset.transform.crops_freq_teacher                                       
            crops_freq_student = data_loader.dataset.transform.crops_freq_student + \
                                        data_loader_aux.dataset.transform.crops_freq_student 
            images_pos = images[:crops_freq_student[0]]    # postives
            images_neg = images[crops_freq_student[0]:]    # in-dist negatives (e.g. rotated view of pos sample) 
            teacher_output = teacher(images_pos[:crops_freq_teacher[0]] \
                                     + images_neg[:crops_freq_teacher[1]] + images_aux[:crops_freq_teacher[2]])          
            student_output = student(images_pos + images_neg + images_aux)  
            loss = dino_loss(student_output, teacher_output, crops_freq_student, crops_freq_teacher, epoch)  

            

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLossNegCon(nn.Module):
    def __init__(self, out_dim, batchsize, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, indist_only=False, aux_only=False):
        super().__init__()
        self.student_temp = student_temp
        self.probs_temp = 0.1
        self.center_momentum = center_momentum
        self.probs_momentum = 0.998
        self.batchsize = batchsize
        
        self.indist_only = indist_only
        self.aux_only = aux_only

        if (indist_only==aux_only) and (indist_only is True):
            raise  AssertionError('Both indist_only and aux_only are True. Set at least one to False')
        # combined defaults to True
        self.combined = True if (indist_only==aux_only) and (indist_only is False) else False
        
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, crops_freq_student, crops_freq_teacher, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # total number of crops
        n_crops_student = len(student_output) // self.batchsize
        n_crops_teacher = len(teacher_output) // self.batchsize

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(n_crops_student)
        temp = self.teacher_temp_schedule[epoch]
        teacher_probs = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_probs.detach().chunk(n_crops_teacher)
        out_dim = teacher_out[0].shape[-1]
        total_loss = 0.
        n_loss_terms = 0
        end_idx_s = torch.cumsum(torch.tensor(crops_freq_student), 0)
        for t in range(crops_freq_teacher[0]):  # loop over pos teacher crops
            start_s = 0
            for k, end_s in enumerate(end_idx_s):  # loop over datasets
                for s in range(start_s, end_s):  # loop over student crops
                    # pos loss
                    if k == 0:
                        if s == t:  # we skip cases where student and teacher operate on the same view
                            continue
                        loss = torch.sum(-teacher_out[t] * F.log_softmax(student_out[s], dim=-1), dim=-1)
                    else:
                        try:
                            if self.indist_only and k == 1:
                                # in-dist only neg loss
                                loss = 1.0 / out_dim * torch.sum(-F.log_softmax(student_out[s], dim=-1), dim=-1)
                            elif self.aux_only and k == 2:
                                loss = 1.0 / out_dim * torch.sum(-F.log_softmax(student_out[s], dim=-1), dim=-1)

                            # both indist and aux here (deafult behaviour)
                            elif self.combined:
                                loss = 0.5 / out_dim * torch.sum(-F.log_softmax(student_out[s], dim=-1), dim=-1)
                            else:
                                continue
                        except:
                            continue
                    
                    total_loss += len(crops_freq_student) * loss.mean()  # scaling loss with batchsize
                    n_loss_terms += 1
                    
                start_s = end_s
        total_loss /= n_loss_terms
        self.center = self.update_ema(teacher_output[:crops_freq_teacher[0]], self.center, self.center_momentum)
        return total_loss

    @torch.no_grad()
    def update_ema(self, output, ema, momentum):
        """
        Update exponential moving aveages for teacher output.
        """
        batch_center = torch.sum(output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(output) * dist.get_world_size())

        # ema update
        return ema * momentum + batch_center * (1 - momentum)

        

    
class DataAugmentation_Contrast(object):
    def __init__(self, global_crops_scale, local_crops_scale, 
                 local_crops_number, image_size, vit_image_size, aux=False): 
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

        if args.neg_aug == 'rot':
            img_neg_aug = d_utils.rotate
            t_neg_aug = d_utils.Identity()
        elif args.neg_aug == 'perm4':
            img_neg_aug = d_utils.Identity()
            t_neg_aug = d_utils.Perm4()            
        elif args.neg_aug == 'perm16':
            img_neg_aug = d_utils.Identity()
            t_neg_aug = d_utils.Perm16()            
        elif args.neg_aug == 'pixel_perm':
            img_neg_aug = d_utils.Identity()
            t_neg_aug = d_utils.RandomPixelPerm()
        elif args.neg_aug == 'rot_360':
            img_neg_aug = transforms.RandomRotation((0,360))
            t_neg_aug = d_utils.Identity()
        elif args.neg_aug == 'img_n':
            img_neg_aug = d_utils.Identity()
            t_neg_aug = d_utils.Identity()
        else:
            raise NotImplemented 

        
        
        # no crop
        self.no_transfo = transforms.Compose([
            transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            normalize,
        ])      

        # neg no crop
        self.no_transfo_neg = transforms.Compose([
            transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            img_neg_aug,
            normalize,
            t_neg_aug
        ])      
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            d_utils.GaussianBlur(1.0, image_size=vit_image_size),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            d_utils.GaussianBlur(0.1, image_size=vit_image_size),
            d_utils.Solarization(0.2),
            normalize,
        ])        
        # neg first global crop
        self.global_transfo1_neg = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            img_neg_aug,
            flip_and_color_jitter,
            d_utils.GaussianBlur(1.0, image_size=vit_image_size),
            normalize,
            t_neg_aug
        ])
        # neg second global crop
        self.global_transfo2_neg = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            img_neg_aug,
            flip_and_color_jitter,
            d_utils.GaussianBlur(0.1, image_size=vit_image_size),
            d_utils.Solarization(0.2),
            normalize,
            t_neg_aug
        ])
        # transformation for the local small crops
        self.local_transfo = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size//2, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            d_utils.GaussianBlur(p=0.5, image_size=vit_image_size),
            normalize,
        ])       
        # neg transformation for the local small crops
        self.local_transfo_neg = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.Resize((vit_image_size, vit_image_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(vit_image_size//2, scale=local_crops_scale, interpolation=Image.BICUBIC),
            img_neg_aug,
            #TF.vflip,
            flip_and_color_jitter,
            d_utils.GaussianBlur(p=0.5, image_size=vit_image_size),
            normalize,
            t_neg_aug
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


                                          

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
