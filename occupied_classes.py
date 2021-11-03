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

import argparse
import torch.backends.cudnn as cudnn
import utils

from occ_classes_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Occupied Classes')
    parser.add_argument('--batch_size_per_gpu', default=200, type=int, help='Per-GPU batch-size')
    parser.add_argument('--local_view', default=False, type=bool, )
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--vit_image_size', type=int, default=256, help="""image size that enters vit; 
        must match with patch_size: num_patches = (vit_image_size/patch_size)**2""")
    parser.add_argument('--image_size', type=int, default=32, help="""image size that enters vit; 
        must match with patch_size: num_patches = (vit_image_size/patch_size)**2""")
    parser.add_argument('--crops_scale', type=float, nargs='+', default=(0.9, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image. Used for large global view cropping.""")
    parser.add_argument('--crops_number', type=int, default=1,
                        help="""Number of local views to generate. Set this parameter to 0 to disable multi-crop training.""")

    parser.add_argument('--pretrained_weights', default='/home/shared/OOD/checkpoints/cifar10/No_Negs/out_0.08_0.09_500/checkpoint.pth',
                        type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--pretrained_out_dim', default=4096,
                        type=int, help="Pretrained DINO classes used on ssl pretraining.")
    parser.add_argument('--use_bn_in_head', default=False, type=bool)
    parser.add_argument('--norm_last_layer', default=False, type=bool)

    parser.add_argument('--load', default=True,
                        type=utils.bool_flag,
                        help="use the saved features of the last checkpoint")
    parser.add_argument('--use_cuda', default=False,
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
    parser.add_argument('--data_path', default='./out_eval/', type=str)
    parser.add_argument('--train_dataset', default='cifar10', type=str)
    args = parser.parse_args()

    find_occ_classes(args, num_crop=1, load=True)
    


