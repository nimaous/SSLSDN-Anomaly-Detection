import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from eval_function import *
from main_train import get_args_parser
import utils 

if __name__ == '__main__':
    
    get_args_parser()
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    args.batch_size_per_gpu=10
    args.use_cuda = False
    utils.init_distributed_mode(args)

    eval_routine(args=args, train_dataset='cifar100', 
            pretrained_weights='/home/shared/OOD/rot_pred_4_classes/out_cifar100_pred_rot4_wrot1_1_neg_glob_views_tt_0.04/checkpoint.pth', 
            overwrite_args=True)


