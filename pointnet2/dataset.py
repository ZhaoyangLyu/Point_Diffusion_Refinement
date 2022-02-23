import torch
from torchvision import transforms
import torch.utils.data as data

import numpy as np

# import data.data_utils as d_utils
# from data.ModelNet40Loader import ModelNet40Cls
# from shapenet_dataloader.shapenet import get_shapenet_dataloader
# from shapenet_dataloader.shapenet_pytorch import get_shapenet_pytorch_dataloader

# from mvp40_dataloader.mvp40_dataset import MVP40H5
from mvp_dataloader.mvp_dataset import ShapeNetH5
# from cascade_dataloader.cascade_dataset import CascadeH5
# from shapenet_chunk_dataloader.shapenet_chunk_dataset import ShapeNetChunkH5
# from partnet_dataloader.partnet_dataset import PartNetH5

# from pointflow_shapenet_loader.args import ShapeNetDatasetArgs
# from pointflow_shapenet_loader.dataloader import get_datasets

def get_dataloader(args, phase='train', rank=0, world_size=1, random_subsample=False, num_samples=0,
                    append_samples_to_last_rank=True):
    
    if num_samples == 'all':
        random_subsample = False
    
    if args['dataset'] == 'mvp_dataset':
        if phase=='train':
            train = True
            shuffle = True
            batch_size=args['batch_size']
            augmentation = args.get('augmentation', False)
            randomly_select_generated_samples = args.get('randomly_select_generated_samples', False)
        else:
            assert phase in ['val', 'test', 'test_trainset']
            # val and test are the same test set for mvp and mvp40 dataset
            # val is used during traing, test is used during generating training data for a trained diffusion model
            train = False
            shuffle = False
            batch_size=args['eval_batch_size']
            augmentation = False
            randomly_select_generated_samples = False # for test set, we don't generate multiple trials
            if phase == 'test_trainset':
                # in this case, we test the trained model's performance on the training set
                train = True
                randomly_select_generated_samples = args.get('randomly_select_generated_samples', False)
        include_generated_samples = args.get('include_generated_samples', False)
        generated_sample_path = args.get('generated_sample_path', '')
        use_mirrored_partial_input = args.get('use_mirrored_partial_input', False)
        number_partial_points = args.get('number_partial_points', 2048)
        load_pre_computed_XT = args.get('load_pre_computed_XT', False) 
        T_step = args.get('T_step', 100)
        XT_folder = args.get('XT_folder', None)
        return_augmentation_params = args.get('return_augmentation_params', False)

        if args.get('augment_data_during_generation', False):
            augmentation = args.get('augmentation', False)

        dataset = ShapeNetH5(args['data_dir'], train=train, npoints=args['npoints'], novel_input=args['novel_input'], 
                    novel_input_only=args['novel_input_only'], scale=args['scale'], 
                    rank=rank, world_size=world_size, random_subsample=random_subsample, num_samples=num_samples,
                    augmentation=augmentation, include_generated_samples=include_generated_samples, 
                    generated_sample_path=generated_sample_path,
                    randomly_select_generated_samples=randomly_select_generated_samples,
                    use_mirrored_partial_input=use_mirrored_partial_input,
                    number_partial_points=number_partial_points,
                    load_pre_computed_XT=load_pre_computed_XT, T_step=T_step, XT_folder=XT_folder,
                    append_samples_to_last_rank=append_samples_to_last_rank,
                    return_augmentation_params=return_augmentation_params)
        
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                                    num_workers=args['num_workers'])

    else:
        raise Exception(args['dataset'], 'dataset is not supported')

    return trainloader



