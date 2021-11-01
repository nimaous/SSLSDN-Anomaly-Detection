import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from eval_function import *

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

    parser.add_argument('--pretrained_weights', default='', 
                        type=str, help="Path to pretrained weights to evaluate.")
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
    parser.add_argument('--data_path', default='/home/shared/DataSets/', type=str)
    parser.add_argument('--train_dataset', default='cifar10', type=str)
    parser.add_argument('--extra_tag', default='_', type=str)

    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    dataset_list_old = ['texture', 'cifar100', 'svhn', 'imagenet30', 'lsun', 
                    'tiny_imagenet', 'cifar10',  'stl10', 'places365', 'places365_b']

    dataset_list = ['cifar10', 'cifar100', 'svhn', 'imagenet30', 'tiny_imagenet', 'lsun', 
                    'tiny_imagenet',  'stl10', 'places365', 'places365_b', 'texture']

    num_crops_list = [1] 

    num_neighbour_list = [-1]
    temperature_list = [0.04] 
    

    ood_result_lst = []
    knn_result_lst = []
    with open(f'{args.train_dataset}_results.txt', 'a+') as file:
        file.write(f'\n\n\n################# Checkpoint: {args.pretrained_weights} ################\n\n')
        file.writelines([f'{key}: {value}\n' for key, value in vars(args).items()])
        file.write('\n\n\n')
        chkpt_name = args.train_dataset + args.extra_tag
        chkpt_path = args.pretrained_weights
        in_ds = args.train_dataset
        knn_dict = {'chkpt_name': chkpt_name, 'train_ds': in_ds}
        print(f"#########. Creating Pipline for {in_ds} ")
        model = load_model(args, chkpt_path)
        for num_crop in num_crops_list:
            train_features, train_labels = extract_feature_pipeline(args, model, in_ds, 
                                                      train=True, crops_number=num_crop)

            if args.reduce_train:
                print(f"######## Applying Kmeans with {args.num_clusters} clusters on Training Features")
                _, train_features = kmeans(X=ftrain, 
                                    num_clusters=args.num_clusters, 
                                    distance='cosine', 
                                    device=torch.device('cuda'))
                print(f"Trean Feature Size After Clustering {train_features.size()}")
            test_features, test_labels = extract_feature_pipeline(args, model, in_ds, 
                                                    train=False)

            if num_crop == 1:
                file.write("___________KNN Results__________________________________\n\n")
                for k in [10, 20]: 
                    if in_ds in ['cifar10', 'svhn']:
                        n_cls = 10
                    elif in_ds in ['cifar100']:
                        n_cls = 100
                    elif in_ds in ['imagenet30']:
                        n_cls = 30                            
                    else:
                        raise NotImplemented                             
                    top1, top5 = knn_classifier(train_features,train_labels,
                                                test_features, test_labels, 
                                                k, args.knn_temperature, 
                                                num_classes=n_cls)
                    knn_dict[f'{k}NN_Top1'] = top1
                    knn_dict[f'{k}NN_Top5'] = top5
                    file.write(f'{k}NN_Top1 \t\t\t {top1}\n')
                knn_result_lst.append(knn_dict)
                
            file.write("\n\n___________OOD Results__________________________________\n\n")  
            file.write("\t\t In \t\t  OOD(Num Samples)  \t\t\t   AUROC \n")
            for ood_ds in dataset_list:
                if ood_ds == in_ds:
                    continue
                ood_features, _ = extract_feature_pipeline(args, model, ood_ds, train=False)    
                for k in num_neighbour_list:
                    for T in temperature_list:
                        conf_dict = {}
                        print(f"\n\n ######## Calculating AUROC for {ood_ds} || Num Neighbour {k} and T {T} ######\n\n")
                        scores_in = OOD_classifier(train_features, test_features, k, T, num_crop, args)
                        scores_out = OOD_classifier(train_features, ood_features, k, T, num_crop, args)
                        labels = torch.cat((torch.ones(scores_in.size(0)), 
                                            torch.zeros(scores_out.size(0))))
                        scores = torch.cat((scores_in, scores_out))
                        auroc = roc_auc_score(labels.numpy(), scores.cpu().numpy())
                        conf_dict['chkpt_name'] = chkpt_name
                        conf_dict['in_ds'] = in_ds
                        conf_dict['ood_ds'] = ood_ds
                        conf_dict['num_crop'] = num_crop
                        conf_dict['num_neighbour'] = k
                        conf_dict['T'] = T
                        conf_dict['AUROC'] = auroc
                        ood_result_lst.append(conf_dict)
                        print(f"AUROC: {auroc}")
                        file.write(f'\t\t {in_ds} \t\t {ood_ds}({ood_features.size(0)}) \t\t\t\t\t  {auroc} \n')
            file.write('___________________________ Done _______________________________')
                            
    ood_df = pd.DataFrame(ood_result_lst)
    knn_df = pd.DataFrame(knn_result_lst)
    ood_df.to_csv(f'ood_{chkpt_name}_results_df.csv', index=False)
    knn_df.to_csv(f'knn_{chkpt_name}_results_df.csv', index=False)
                            
                            
    dist.barrier()
