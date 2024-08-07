# self-supervised anomaly detection by self-distillation and negative sampling


### Training for CIFAR 10 as in-distribution and ImageNet as auxiliary and Rotation as negative augmentation
```
python -m torch.distributed.launch --master_port 8998 --nproc_per_node=4 main_train.py --arch vit_small --epochs=500 --batch_size_per_gpu=32 --out_dim=4096 --lr=0.004 --warmup_epochs=30 --weight_decay=0.04 --weight_decay_end=0.4 --local_crops_number=8 --local_crops_scale 0.15 0.4 --global_crops_scale 0.4 1. --image_size=32 --vit_image_size=256 --patch_size=16 --use_fp16=false --norm_last_layer=false --teacher_temp=0.01 --warmup_teacher_temp=0.04 --warmup_teacher_temp_epochs=500  --output_dir ../cifar10  --in_dist=cifar10
```

### Eval
```
python -m torch.distributed.launch --master_port 8993 --nproc_per_node=1 eval_checkpoints.py --train_dataset=imagenet30 --pretrained_weights=../imagnet30_pred_rot4_wrot1_1_neg_glob_views_tt_0.04/checkpoint.pth   --extra_tag=rotpred_exp1 


```







<!-- ### Visualize occupied classes
check args first ...
```
python -m occupied_classes
``` -->


### Cite


```
@inproceedings{rafiee2022self,
  title={Self-Supervised Anomaly Detection by Self-Distillation and Negative Sampling},
  author={Rafiee, Nima and Gholamipoor, Rahil and Adaloglou, Nikolas and Jaxy, Simon and Ramakers, Julius and Kollmann, Markus},
  booktitle={Artificial Neural Networks and Machine Learning--ICANN 2022: 31st International Conference on Artificial Neural Networks, Bristol, UK, September 6--9, 2022, Proceedings; Part IV},
  pages={459--470},
  year={2022},
  organization={Springer Nature Switzerland Cham}
}
```
