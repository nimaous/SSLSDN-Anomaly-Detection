# DINO_AnoDetect


Best results so far
```
python -m torch.distributed.launch --master_port 8997 --nproc_per_node=4 main_dino_aux_v3.py --arch vit_small --epochs=500 --batch_size_per_gpu=32 --out_dim=4096 --lr=0.004 --warmup_epochs=30 --weight_decay=0.04 --weight_decay_end=0.4 --local_crops_number=8 --local_crops_scale 0.15 0.4 --global_crops_scale 0.4 1. --image_size=32 --vit_image_size=256 --patch_size=16 --use_fp16=false --norm_last_layer=false --teacher_temp=0.01 --warmup_teacher_temp=0.055 --warmup_teacher_temp_epochs=500 --warmup_contrast_temp=0.25 --contrast_temp=0.25 --warmup_contrast_temp_epochs=50  --output_dir ./out  &
```

test command

```
python -m torch.distributed.launch --master_port 8992 --nproc_per_node=1 main_train_pred_rot.py --arch vit_small --epochs=10 --batch_size_per_gpu=3 --out_dim=4096 --lr=0.004 --warmup_epochs=5 --weight_decay=0.04 --weight_decay_end=0.4 --local_crops_number=2 --local_crops_scale 0.15 0.4 --patch_size=4 --vit_image_size=64  --output_dir ./test_out 
```




