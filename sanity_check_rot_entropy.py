import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
import torch.backends.cudnn as cudnn
import utils
import vision_transformer as vits

from vision_transformer import DINOHead, RotationHead

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

    parser.add_argument('--pretrained_weights', default='/home/shared/OOD/rot_pred/rot_pred_4_classes/checkpoint.pth', 
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
    parser.add_argument('--data_path', default='/home/shared/DataSets/', type=str)
    parser.add_argument('--train_dataset', default='cifar10', type=str)
    parser.add_argument('--extra_tag', default='_', type=str)

    args = parser.parse_args()
    args.out_dim = 4096

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    

    dataset_list = ['cifar10', 'cifar100', 'svhn', 'imagenet30', 'tiny_imagenet', 'lsun', 
                    'tiny_imagenet',  'stl10', 'places365', 'places365_b', 'texture']

    num_neighbour_list = [-1]
    temperature_list = [0.04] 
    


    chkpt_name = args.train_dataset + args.extra_tag
    chkpt_path = args.pretrained_weights
    in_ds = args.train_dataset
    knn_dict = {'chkpt_name': chkpt_name, 'train_ds': in_ds}


    # model 
    teacher = vits.__dict__[args.arch](
            img_size=[args.vit_image_size],
            patch_size=args.patch_size,
        )
    embed_dim = teacher.embed_dim

    rot_head = RotationHead(
        embed_dim,
        4, # out dim - 4 classes
    )
    main_head = DINOHead(
        embed_dim,
        4096,
        use_bn=False,
        norm_last_layer=False)

    model = utils.MultiCropWrapper( teacher, main_head)

    model = utils.load_pretrained_weights(model, args.pretrained_weights, 
                                  args.checkpoint_key, 
                                  args.arch, args.patch_size,
                                  remove_head=False)
    print('model is loaded')
    # TODO
    # train_features, train_labels = extract_feature_pipeline(args, model, in_ds, 
    #                                             train=True, crops_number=1)

    
    # test_features, test_labels = extract_feature_pipeline(args, model, in_ds, 
    #                                         train=False)

                            
