import os
import sys, io
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

import pickle

from util import rescale, find_max_epoch, print_size, sampling, calc_diffusion_hyperparams, AverageMeter
from models.pointnet2_ssg_sem import PointNet2SemSegSSG
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from dataset import get_dataloader
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict

# from pytorch3d.loss.chamfer import chamfer_distance
# from shapenet_dataloader.emd import EMD_distance
# from chamfer_loss import Chamfer_Loss
# from eval.plot_result import plot_result
# from eval.compare_eval_result import plot_result_list
from completion_eval import evaluate

import pdb

# from dataparallel import MyDataParallel
import h5py
from shutil import copyfile, rmtree

# shapenet categories
name_to_number ={
'plane': '02691156',
# 'bench': '02828884',
'cabinet': '02933112',
'car': '02958343',
'chair': '03001627',
# 'monitor': '03211117',
'lamp': '03636649',
# 'speaker': '03691459',
# 'firearm': '04090263',
'couch': '04256520',
'table': '04379243',
# 'cellphone': '04401088',
'watercraft': '04530566'}

number_to_name = {}
for k in name_to_number.keys():
    number_to_name[name_to_number[k]] = k

def main(config_file, pointnet_config, trainset_config, train_config, diffusion_config, diffusion_hyperparams, 
            batch_size, ckpt_path, ckpt_iter, phase, rank=0, world_size=1, dataset='shapenet', 
            std_out_file='generation.log', trial_index=None, save_multiple_t_slices=False,
            t_slices=[5, 10, 20, 50, 100, 200, 400, 600, 800],
            use_a_precomputed_XT=False, T_step=100,
            refine_config = None, ckpt_name=None, num_points=None, parallel=True,
            augment_data_during_generation=False,
            generate_samples_for_a_subset_of_the_datset=False,
            subset_indices_file=None,
            manually_specified_save_dir='',
            fast_sampling=False,
            fast_sampling_config=None):
    """
    Generate 3D point clouds from the trained model

    Parameters:
    batch_size (int):              number of samples to generate, default is 4
    ckpt_path (str):                checkpoint path
    ckpt_iter (int or 'max', or 'best):       the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum iteration if 'max' is selected
    """
    if save_multiple_t_slices:
        print('multiple slices will be saved at t steps', t_slices)
    assert dataset in ['shapenet', 'shapenet_pytorch', 'mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']
    root_directory = train_config['root_directory']
    # generate experiment (local) path
    local_path = "T{}_betaT{}".format(diffusion_config["T"], diffusion_config["beta_T"])
    local_path = local_path + '_' + pointnet_config['model_name']

    if train_config.get('task', 'completion') == 'refine_completion':
        exp_name_split = list(os.path.split(refine_config['exp_name']))
        exp_name_split[-1] = 'refine_exp_' + exp_name_split[-1]
        exp_name = os.path.join(*exp_name_split)
        local_path = os.path.join(local_path, exp_name)

    # find ckpt point path
    ckpt_path = os.path.join(root_directory, local_path, ckpt_path)
    if ckpt_name is None or len(ckpt_name)==0:
        if ckpt_iter == 'max' or ckpt_iter == 'best':
            ckpt_iter = find_max_epoch(ckpt_path, 'pointnet_ckpt', mode=ckpt_iter)
        else:
            ckpt_iter = int(ckpt_iter)
        model_path = os.path.join(ckpt_path, 'pointnet_ckpt_{}.pkl'.format(ckpt_iter))
    else:
        model_path = os.path.join(ckpt_path, ckpt_name)

    if dataset in ['shapenet', 'shapenet_pytorch']:
        save_dir = 'shapenet_dataloader/data/shapenet/generated_samples'
    elif dataset == 'mvp_dataset':
        save_dir = 'mvp_dataloader/data/mvp_dataset/generated_samples'
    elif dataset == 'shapenet_chunk':
        save_dir = 'shapenet_chunk_dataloader/data/shapenet_chunk/generated_samples'
    elif dataset == 'mvp40':
        save_dir = 'mvp40_dataloader/data/mvp40/generated_samples'
    elif dataset == 'partnet':
        save_dir = 'partnet_dataloader/data/partnet/generated_samples'
    else:
        raise Exception('%s dataset is not suported' % dataset)
    
    save_dir = os.path.join(save_dir, local_path)
    if ckpt_name is None or len(ckpt_name)==0:
        save_dir = os.path.join(save_dir, 'ckpt_%d' % ckpt_iter)
    else:
        save_dir = os.path.join(save_dir, ckpt_name.split('.')[0])

    if fast_sampling:
        save_dir = os.path.join(save_dir, 'fast_sampling')
        directory = 'fast_sampling_config'
        for key in fast_sampling_config.keys():
            directory = directory + '_' + key + '_' + str(fast_sampling_config[key])
        save_dir = os.path.join(save_dir, directory)
    
    if trial_index is not None:
        save_dir = os.path.join(save_dir, 'trial_%d' % trial_index)
    
    if len(manually_specified_save_dir) > 0:
        save_dir = manually_specified_save_dir

    os.makedirs(save_dir, exist_ok=True)
    
    try:
        copyfile(config_file, os.path.join(save_dir, os.path.split(config_file)[1]))
    except:
        print('The two files are the same, donot need to copy')

    # phase = test_trainset, test,          benchmark for mvp dataset and mvp40 dataset
    # phase = test_trainset, cascade_test,  test, val for shapenet_chunk
    if phase == 'test_trainset':
        save_dir = os.path.join(save_dir, 'train')
    # elif phase == 'val':
    #     save_dir = os.path.join(save_dir, 'val')
    elif phase == 'test':
        save_dir = os.path.join(save_dir, 'test')
    # elif phase == 'benchmark':
    #     save_dir = os.path.join(save_dir, 'benchmark')
    # elif phase == 'cascade_test':
    #     save_dir = os.path.join(save_dir, 'cascade_test')
    else:
        raise Exception('phase %s is not supported' % phase)

    sys.stdout = open(std_out_file, 'a')

    if world_size > 1:
        save_dir = os.path.join(save_dir, 'rank_%d' % rank)

    if dataset in ['shapenet', 'shapenet_pytorch']:
        for key in number_to_name.keys():
            os.makedirs(os.path.join(save_dir, key), exist_ok=True)
    elif dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
        os.makedirs(save_dir, exist_ok=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # build model
    if train_config["conditioned_on_cloud"]:
        net = PointNet2CloudCondition(pointnet_config).cuda()
    else:
        net = PointNet2SemSegSSG(pointnet_config).cuda()
    print_size(net)

    
    # load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Model from %s has been trained for %s seconds' % (os.path.split(model_path)[-1], checkpoint['training_time_seconds']))
    except:
        raise Exception('Model is not loaded successfully')
    
    # inference
    # noise_reduce_factor = train_config['noise_reduce_factor']
    
    # get data loader
    # assert train_config['dataset'] in ['shapenet', 'shapenet_pytorch', 'mvp_dataset', 'shapenet_chunk']
    trainset_config['batch_size'] = batch_size * torch.cuda.device_count()
    trainset_config['eval_batch_size'] = batch_size * torch.cuda.device_count()
    testloader = get_dataloader(trainset_config, phase=phase, rank=rank, world_size=world_size,
                    append_samples_to_last_rank=False) 
                    # random_subsample=generate_samples_for_a_subset_of_the_datset, num_samples=1000
                    # use_a_random_indices_file=generate_samples_for_a_subset_of_the_datset,
                    # random_indices_file=subset_indices_file)
        
    # pdb.set_trace()
    if dataset in ['shapenet', 'shapenet_pytorch']:
        data_scale = trainset_config['scale_factor'] 
    elif dataset in ['mvp_dataset', 'shapenet_chunk', 'mvp40', 'partnet']:
        data_scale = trainset_config['scale'] 

    if (dataset=='mvp_dataset' and phase == 'benchmark') or (dataset=='shapenet_chunk' and phase=='test'):
        # do not compute emd and cd because there is no groud truth shape
        compute_emd = False
        compute_cd = False
    else:
        compute_emd = True
        compute_cd = True

    if num_points is not None and num_points > 2048:
        # emd computation is too memory consuming, we only compute emd when the number of points
        # in the gt point cloud is 2048
        compute_emd = False

    if train_config.get('task', 'completion') == 'refine_completion':
        refine_output_scale_factor = refine_config['output_scale_factor']
    else:
        refine_output_scale_factor = None

    point_upsample_factor = pointnet_config.get('point_upsample_factor', 1) 
    include_displacement_center_to_final_output = pointnet_config.get('include_displacement_center_to_final_output', False) 
    
    CD_loss, EMD_loss, total_meta, metrics = evaluate(net, testloader, diffusion_hyperparams, 
        print_every_n_steps=diffusion_config["T"] // 5, 
        parallel=parallel, dataset=trainset_config['dataset'], scale=data_scale, 
        save_generated_samples=True, save_dir = save_dir,
        save_multiple_t_slices=save_multiple_t_slices,
        t_slices=t_slices,
        use_a_precomputed_XT=use_a_precomputed_XT, T_step=T_step,
        compute_emd=compute_emd, compute_cd=compute_cd,
        task=train_config.get('task', 'completion'),
        num_points=num_points, refine_output_scale_factor=refine_output_scale_factor,
        augment_data_during_generation=augment_data_during_generation,
        point_upsample_factor=point_upsample_factor,
        include_displacement_center_to_final_output=include_displacement_center_to_final_output,
        fast_sampling=fast_sampling,
        fast_sampling_config=fast_sampling_config, diffusion_config=diffusion_config,
        return_all_metrics=True)
    
    if ckpt_name is None or len(ckpt_name)==0:
        save_file = os.path.join(save_dir, 'eval_result_ckpt_%d.pkl' % (ckpt_iter))
    else:
        save_file = os.path.join(save_dir, 'eval_result_%s.pkl' % (ckpt_name.split('.')[0]))
        ckpt_iter = ckpt_name.split('.')[0]
    # save_file = os.path.join(save_dir, 'eval_result_ckpt_%d.pkl' % (ckpt_iter))
    handle = open(save_file, 'wb')
    pickle.dump({'meta':total_meta, 'cd_distance':metrics['cd_distance'].detach().cpu().numpy(), 
                    'emd_distance':metrics['emd_distance'].detach().cpu().numpy(),
                    'f1':metrics['f1'].detach().cpu().numpy(),
                    'avg_cd':CD_loss, 'avg_emd':EMD_loss, 'iter':ckpt_iter}, handle)
    handle.close()
    if ckpt_name is None or len(ckpt_name)==0:
        print('have saved eval result at iter %d to %s' % (ckpt_iter, save_file))
        print("iteration: {} \tCD loss: {} \tEMD loss: {} \tF1 Score: {}".format(ckpt_iter, CD_loss, EMD_loss, 
                                                        metrics['f1'].detach().cpu().mean()), flush=True)
    else:
        print('have saved eval result at %s to %s' % (ckpt_name, save_file))
        print("Ckpt name: {} \tCD loss: {} \tEMD loss: {} \tF1 Score: {}".format(ckpt_name, CD_loss, EMD_loss,
                                                        metrics['f1'].detach().cpu().mean()), flush=True)
    
    try:
        sys.stdout = io.StringIO()
    except:
        pass
    copyfile(std_out_file, os.path.join(save_dir, os.path.split(std_out_file)[1] ))
    return CD_loss, EMD_loss


if __name__ == "__main__":
    '''
    running examples:
    python generate_samples.py --config exp_mvp_dataset_completion/T1000_betaT0.02_shape_completion_avg_max_pooling_mirror_rot_90_scale_1.2_translation_0.2/logs/checkpoint/config_avg_max_pooling_mirror_rot_90_scale_1.2_translation_0.2.json --ckpt_iter 545999 --phase test_trainset --use_a_precomputed_XT --num_trials 20
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='number of points in each shape')
    parser.add_argument('--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max" or "best"')
    parser.add_argument('--ckpt_name', default='',
                        help='Which checkpoint to use, the file name of the ckeckpoint')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='batchsize to generate data')
    parser.add_argument('-p', '--phase', type=str, default='test_trainset',
                        help='which part of the dataset to generated samples')
    parser.add_argument('--save_multiple_t_slices', action='store_true',
                    help='whether to save multiple t slices (default: false)')
    parser.add_argument('--t_slices', type=str, default='[5,10,20,50,100,200,400,600,800]',
                    help='the intermediate t slices to save')

    parser.add_argument('--fast_sampling', action='store_true',
                    help='whether to use fast sampling (default: false)')
    parser.add_argument('--fast_sampling_config', type=str, default='100; var; quadratic; 0.0',
                    help='fast_sampling_config: length; sampling_method; schedule type; kappa')

    parser.add_argument('--save_dir', type=str, default='',
                        help='the directory to save the generated samples')
    # parser.add_argument('--generate_samples_for_a_subset_of_the_datset', action='store_true',
    #                 help='whether to generate_samples_for_a_subset_of_the_datset (default: false)')
    # parser.add_argument('--subset_indices_file', type=str, default='mvp_dataloader/random_indices.pkl',
    #                     help='indices of the samples that we want to generate complete point clouds')

    parser.add_argument('--augment_data_during_generation', action='store_true',
                    help='whether to augment data during evaluation (default: false)')
    parser.add_argument('--augmentation_during_generation', type=str, default='1.2; 60; 0.5; 0.05',
                        help='augmentations during generation, (scale; rotation; mirror; translation)')

    parser.add_argument('--use_a_precomputed_XT', action='store_true',
                    help='whether to use precomputed XT to generate samples (default: false)')
    parser.add_argument('--T_step', type=int, default=100,
                        help='the t step to reverse begin with')
    # load_pre_computed_XT=False, T_step=100, XT_folder=None
    parser.add_argument('--XT_folder', type=str, default='mvp_dataloader/data/mvp_dataset/generated_samples/T1000_betaT0.02_shape_completion_avg_max_pooling_mirror_rot_90_scale_1.2_translation_0.2/ckpt_545999',
                        help='the folder that stores the precomputed XT')

    parser.add_argument('--parallel', action='store_true',
                    help='whether to use all visible gpus (default: false)')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='the rank of the splits of the dataset')
    parser.add_argument('-w', '--world_size', type=int, default=1,
                        help='number of splits of the dataset')
    parser.add_argument('-d', '--device_ids', type=str, default='0,1,2,3,4,5,6,7',
                        help='gpu device indices to use')
    parser.add_argument('-s', '--std_out_file', type=str, default='generation.log',
                        help='generation log output file')

    # multiple trials settings
    parser.add_argument('-n', '--num_trials', type=int, default=1,
                        help='number of trials to generate for each partial point cloud')
    parser.add_argument('--start_trial', type=int, default=1,
                        help='trial index to start')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)
    gen_config              = config["gen_config"]
    pointnet_config         = config["pointnet_config"]     # to define pointnet
    diffusion_config        = config["diffusion_config"]    # basic hyperparameters
    train_config            = config["train_config"]

    if train_config.get('task', 'completion') == 'refine_completion':
        refine_config = config['refine_config']
        assert not args.save_multiple_t_slices
        assert not args.use_a_precomputed_XT
    else:
        refine_config = None
    
    if train_config['dataset'] == 'modelnet40':
        trainset_config = config["trainset_config"]     # to load trainset
    elif train_config['dataset'] == 'mvp_dataset':
        trainset_config = config["mvp_dataset_config"]
    elif train_config['dataset'] == 'shapenet':
        trainset_config = config["shapenet_config"]
    elif train_config['dataset'] == 'shapenet_pytorch':
        trainset_config = config["shapenet_pytorch_config"]
    elif train_config['dataset'] == 'pointflow_shapenet':
        trainset_config = config["pointflow_shapenet_config"]
    elif train_config['dataset'] == 'shapenet_chunk':
        trainset_config = config['shapenet_chunk_config']
    elif train_config['dataset'] == 'mvp40':
        trainset_config = config['mvp40_config']
    elif train_config['dataset'] == 'partnet':
        trainset_config = config['partnet_config']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])
    # load_pre_computed_XT=False, T_step=100, XT_folder
    trainset_config['load_pre_computed_XT'] = args.use_a_precomputed_XT
    trainset_config['T_step'] = args.T_step
    trainset_config['XT_folder'] = args.XT_folder

    if args.augment_data_during_generation:
        aug = args.augmentation_during_generation
        aug = aug.replace(' ', '') # remove space
        aug = aug.split(';')
        aug = [eval(au) for au in aug]
        augmentation = {
            "pc_augm_scale": aug[0],
            "pc_augm_rot": True,
            "pc_rot_scale": aug[1],
            "pc_augm_mirror_prob": aug[2],
            "pc_augm_jitter": False,
            "translation_magnitude": aug[3],
            "noise_magnitude_for_generated_samples": 0
        }
        print('We will augment the data during evaluation, and the augmentation is\n', augmentation)
        trainset_config['augmentation'] = augmentation
        trainset_config['augment_data_during_generation'] = True
        trainset_config['return_augmentation_params'] = True

    fast_sampling_config = None
    if args.fast_sampling:
        fast_sampling_config_str = args.fast_sampling_config
        fast_sampling_config_str = fast_sampling_config_str.replace(' ', '') # remove space
        fast_sampling_config_str = fast_sampling_config_str.split(';')
        fast_sampling_config = {}
        fast_sampling_config['length'] = eval(fast_sampling_config_str[0])
        fast_sampling_config['sampling_method'] = fast_sampling_config_str[1]
        fast_sampling_config['schedule'] = fast_sampling_config_str[2]
        fast_sampling_config['kappa'] = eval(fast_sampling_config_str[3])



    # global diffusion_hyperparams
    diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        if args.num_trials <= 1:
            if args.world_size>1:
                if os.path.isdir('generation_logs'):
                    try:
                        rmtree('generation_logs')
                        print('The original log directory generation_logs has been removed')
                    except:
                        pass
                time.sleep(5) # wait all processes to check whether the original log directory exists
                os.makedirs('generation_logs', exist_ok=True)
                std_out_file = args.std_out_file.split('.')[0]
                std_out_file = std_out_file + '_rank_%d.log' % args.rank
                std_out_file = os.path.join('generation_logs', std_out_file)
            else:
                std_out_file = args.std_out_file
                if os.path.isfile(std_out_file):
                    os.remove(std_out_file)
                    print('The original log file %s has been removed' % std_out_file)
            CD_loss, EMD_loss = main(args.config, pointnet_config, trainset_config, train_config, diffusion_config, 
                        diffusion_hyperparams, args.batch_size, gen_config['ckpt_path'], args.ckpt_iter, 
                        args.phase, args.rank, args.world_size, dataset = train_config['dataset'],
                        std_out_file = std_out_file,
                        save_multiple_t_slices=args.save_multiple_t_slices,
                        use_a_precomputed_XT=args.use_a_precomputed_XT, T_step=args.T_step,
                        refine_config=refine_config, ckpt_name=args.ckpt_name, num_points=args.num_points,
                        parallel=args.parallel, augment_data_during_generation=args.augment_data_during_generation,
                        # generate_samples_for_a_subset_of_the_datset=args.generate_samples_for_a_subset_of_the_datset,
                        # subset_indices_file=args.subset_indices_file,
                        manually_specified_save_dir=args.save_dir,
                        fast_sampling=args.fast_sampling,
                        fast_sampling_config=fast_sampling_config,
                        t_slices = eval(args.t_slices))
        else:
            if os.path.isdir('generation_logs'):
                try:
                    rmtree('generation_logs')
                    print('The original log directory generation_logs has been removed')
                except:
                    pass
            time.sleep(5)
            os.makedirs('generation_logs', exist_ok=True)
            current_trial = args.start_trial
            end_trial = args.start_trial + args.num_trials
            for i in range(args.num_trials):
                try:
                    print('generating trial %d [start:%d, end %d]' % (current_trial, args.start_trial, end_trial))
                except:
                    pass
                std_out_file = args.std_out_file.split('.')[0]
                if args.world_size>1:
                    std_out_file = std_out_file + '_trial_%d_rank_%d.log' % (current_trial, args.rank)
                else:
                    std_out_file = std_out_file + '_trial_%d.log' % current_trial
                std_out_file = os.path.join('generation_logs', std_out_file)
                CD_loss, EMD_loss = main(args.config, pointnet_config, trainset_config, train_config, diffusion_config, 
                        diffusion_hyperparams, args.batch_size, gen_config['ckpt_path'], args.ckpt_iter, 
                        args.phase, args.rank, args.world_size, dataset = train_config['dataset'],
                        std_out_file = std_out_file, trial_index=current_trial,
                        save_multiple_t_slices=args.save_multiple_t_slices,
                        use_a_precomputed_XT=args.use_a_precomputed_XT, T_step=args.T_step,
                        refine_config=refine_config, ckpt_name=args.ckpt_name, num_points=args.num_points,
                        parallel=args.parallel, augment_data_during_generation=args.augment_data_during_generation,
                        # generate_samples_for_a_subset_of_the_datset=args.generate_samples_for_a_subset_of_the_datset,
                        # subset_indices_file=args.subset_indices_file,
                        manually_specified_save_dir=args.save_dir,
                        fast_sampling=args.fast_sampling,
                        fast_sampling_config=fast_sampling_config,
                        t_slices = eval(args.t_slices))
                
                current_trial = current_trial + 1
