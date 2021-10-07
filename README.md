# DINO_AnoDetect

test command
```
python -m torch.distributed.launch --master_port 8992 --nproc_per_node=1 main_train.py --arch vit_small --epochs=10 --batch_size_per_gpu=2 --out_dim=4096 --lr=0.004 --warmup_epochs=5 --weight_decay=0.04 --weight_decay_end=0.4 --local_crops_number=2 --local_crops_scale 0.15 0.4 --patch_size=4 --vit_image_size=64  --output_dir ./test_out 
```
