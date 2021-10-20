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

from torch.utils.tensorboard import SummaryWriter


import utils
import vision_transformer as vits
from vision_transformer import DINOHead

from main_train import get_args_parser, DataAugmentation_Contrast

from dino_loss import  DINOLossNegCon, DINOLoss_vanilla 


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


def train_dino(args, writer):
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
        aux=False,
    )

    transform_aux = DataAugmentation_Contrast(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.image_size,
        args.vit_image_size,
        aux=True,
    )

    # root=args.data_path
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # dataset = datasets.ImageFolder(root=args.data_path+'ImageNet30/train/', 
    #                                              transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"In-distriubtion Data loaded: there are {len(dataset)} images.")

    dataset_aux = datasets.ImageFolder(args.data_path,
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
    head_1 = DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )

    rotation_out_dim = 64
    head_2 = DINOHead(
        embed_dim,
        rotation_out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )

    student = MultiCropDoubleHeadWrapper(student, head_1, head_2)
    teacher = MultiCropDoubleHeadWrapper(teacher, head_1, head_2)
    
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss_vanilla(
        args.out_dim,
        args.batch_size_per_gpu,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    dino_loss_neg = DINOLoss_vanilla(
        rotation_out_dim,
        args.batch_size_per_gpu,  
        args.warmup_teacher_temp, 
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
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
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, dino_loss_neg,
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
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, dino_loss_neg, data_loader, data_loader_aux,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    bs = args.batch_size_per_gpu
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
        #images_aux = [im.cuda(non_blocking=True) for im in images_aux]

        
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # number of crops per dataset
            # e.g. crops_freq_teacher = [3,1,1] => 3 x pos , 1 x in-dist neg , 1 x aux neg
            crops_freq_teacher = data_loader.dataset.transform.crops_freq_teacher + \
                                 data_loader_aux.dataset.transform.crops_freq_teacher
            
            crops_freq_student = data_loader.dataset.transform.crops_freq_student + \
                                 data_loader_aux.dataset.transform.crops_freq_student
            
            # positives
            images_pos_student = images[:crops_freq_student[0]]  
            # in-dist negatives (e.g. rotated view of pos sample)
            images_neg_student = images[crops_freq_student[0]:]  

            # concat image lists: 
            images_pos_teacher = images_pos_student[:crops_freq_teacher[0]]
            images_neg_teacher = images_neg_student[:crops_freq_teacher[1]]
            
            # forward to multi-crop wrapper
            teacher_out_1, teacher_out_2 = teacher(images_pos_teacher, images_neg_teacher)
            student_out_1, student_out_2  = student(images_pos_student, images_neg_student)

            loss_dino = dino_loss(student_out_1, teacher_out_1, epoch)
            loss_rot = dino_loss_neg(student_out_2, teacher_out_2, epoch)
            loss = loss_dino + loss_rot

            
        total_loss = loss.item()
        if not math.isfinite(total_loss):
            print("Loss is {}, stopping training".format(total_loss), force=True)
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
        metric_logger.update(loss=total_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if utils.is_main_process():
            writer.add_scalar("Train loss step", total_loss, it)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], it)
            writer.add_scalar("dino loss ", loss_dino.item(), it)
            writer.add_scalar("rotation neg soft CE loss", loss_rot.item(), it)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if utils.is_main_process():
        try:
            writer.add_scalar("Train loss epoch", torch.Tensor([metric_logger.meters['loss'].global_avg]), epoch)
        except:
            sys.exit(1)

    
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class MultiCropDoubleHeadWrapper(nn.Module):
    """
    Supports double head for positives and negatives
    """
    def __init__(self, backbone, head, head_aux = None):
        super(MultiCropDoubleHeadWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.head_aux = head_aux

    def forward_backbone(self,x):
        shapes_sorted, sort_idx = torch.sort(torch.Tensor([inp.shape[-1] for inp in x]))
        idx_crops = torch.cumsum(torch.unique_consecutive(shapes_sorted, return_counts=True)[1], 0)
        start_idx = 0
        output = torch.empty((len(x), len(x[0]), self.backbone.embed_dim)).to(x[0].device)  
        for end_idx in idx_crops:
            batch_idx = sort_idx[start_idx:end_idx]  # The indices of tensors of this shape
            _out = self.backbone(torch.cat([x[i] for i in batch_idx]))   # Batch them together
            
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            _out = torch.stack(_out.chunk(len(batch_idx)))
            output.index_copy_(0, batch_idx.cuda(), _out)
            start_idx = end_idx
        return torch.cat(torch.unbind(output))

    def forward(self, x, y=None):
        # convert to list
        if not isinstance(x, list):
            x = [x]  
        x1 = self.forward_backbone(x)
        x1 = self.head(x1)
        if y is not None:
            if not isinstance(x, list):
                y = [y]
            y1 = self.forward_backbone(y)
            y1 = self.head_aux(y1)
            return x1,y1

        return x1

if __name__ == '__main__':

    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()
    date_time = now.strftime("TBLogs_%m-%d_time_%Hh%M")
    log_folder = os.path.join(args.output_dir, date_time)
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    if utils.is_main_process():
        writer = SummaryWriter(log_folder)

    train_dino(args, writer)