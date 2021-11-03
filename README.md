# DINO_AnoDetect


### Best results so far
```
python -m torch.distributed.launch --master_port 8998 --nproc_per_node=4 main_train_aux_only_360.py --arch vit_small --epochs=500 --batch_size_per_gpu=32 --out_dim=4096 --lr=0.004 --warmup_epochs=30 --weight_decay=0.04 --weight_decay_end=0.4 --local_crops_number=8 --local_crops_scale 0.15 0.4 --global_crops_scale 0.4 1. --image_size=32 --vit_image_size=256 --patch_size=16 --use_fp16=false --norm_last_layer=false --teacher_temp=0.01 --warmup_teacher_temp=0.04 --warmup_teacher_temp_epochs=500  --output_dir ../cifar10_aux_only_rotate360  --in_dist=cifar10


python -m torch.distributed.launch --master_port 9001 --nproc_per_node=1 main_train_pred_rot.py --arch vit_small --epochs=500 --batch_size_per_gpu=2 --out_dim=4096 --lr=0.004 --warmup_epochs=30 --weight_decay=0.04 --weight_decay_end=0.4 --local_crops_number=8 --local_crops_scale 0.15 0.4 --global_crops_scale 0.4 1. --image_size=32 --vit_image_size=256 --patch_size=16 --use_fp16=false --norm_last_layer=false --teacher_temp=0.01 --warmup_teacher_temp=0.04 --warmup_teacher_temp_epochs=500  --output_dir ./out_test_code --in_dist=cifar10
```

#### CIFAR100 + translation 30%, warmup_teacher_temp=0.04
```
python -m torch.distributed.launch --master_port 8998 --nproc_per_node=4 main_train_translation.py --arch vit_small --epochs=500 --batch_size_per_gpu=32 --out_dim=4096 --lr=0.004 --warmup_epochs=30 --weight_decay=0.04 --weight_decay_end=0.4 --local_crops_number=8 --local_crops_scale 0.15 0.4 --global_crops_scale 0.4 1. --image_size=32 --vit_image_size=256 --patch_size=16 --use_fp16=false --norm_last_layer=false --teacher_temp=0.01 --warmup_teacher_temp=0.055  --warmup_teacher_temp_epochs=500  --output_dir ./out_cifar100  --in_dist=cifar100
```

### Eval
```
python -m torch.distributed.launch --master_port 8993 --nproc_per_node=1 eval_checkpoints.py --train_dataset=imagenet30 --pretrained_weights=../imagnet30_pred_rot4_wrot1_1_neg_glob_views_tt_0.04/checkpoint.pth   --extra_tag=rotpred_exp1 


```







### Visualize occupied classes
check args first ...
```
python -m occupied_classes
```



### test command

```
python -m torch.distributed.launch --master_port 8991 --nproc_per_node=1 main_train_aux_only_360.py --arch vit_small --epochs=10 --batch_size_per_gpu=3 --out_dim=4096 --lr=0.004 --warmup_epochs=5 --weight_decay=0.04 --weight_decay_end=0.4 --local_crops_number=2 --local_crops_scale 0.15 0.4 --patch_size=4 --vit_image_size=64  --output_dir ./test_out --in_dist=cifar10
```



